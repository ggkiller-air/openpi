import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PartialCheckpointWeightLoader(WeightLoader):
    """Loads a checkpoint but reinitializes the action-head projections.

    Use this when finetuning a base model with a DIFFERENT ``action_dim`` than the base was
    trained with (e.g. SONIC needs action_dim=78 vs the base's 32). The VLM + action-expert
    backbone is loaded from the checkpoint, while the layers whose shape is tied to
    ``action_dim`` (``action_in_proj``, ``action_out_proj``, and ``state_proj`` for non-pi05)
    are left at their fresh initialization.

    Mechanism: any loaded param that matches ``skip_regex`` OR whose shape differs from the
    fresh reference is dropped before merging; ``_merge_params`` then re-fills those keys from
    the reference (fresh ``ShapeDtypeStruct``), which passes ``check_pytree_equality`` and is
    subsequently left at init by the training loop.
    """

    params_path: str
    skip_regex: str = r"(.*/)?(action_in_proj|action_out_proj|state_proj)/.*"

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        pattern = re.compile(self.skip_regex)
        dropped = []
        for k in list(flat_loaded):
            ref = flat_ref.get(k)
            shape_mismatch = ref is not None and getattr(flat_loaded[k], "shape", None) != getattr(ref, "shape", None)
            if pattern.fullmatch(k) or shape_mismatch:
                del flat_loaded[k]
                dropped.append(k)
        if dropped:
            logger.info("PartialCheckpointWeightLoader: reinitializing %d param(s) from fresh init: %s", len(dropped), sorted(dropped))

        loaded_params = flax.traverse_util.unflatten_dict(flat_loaded, sep="/")
        # Re-fill the dropped projections (and any LoRA) from the fresh reference params.
        return _merge_params(loaded_params, params, missing_regex=r".*lora.*|" + self.skip_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
