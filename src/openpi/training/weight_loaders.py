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

    Mechanism: iterate over the FRESH reference keys. A key is left at its fresh init when it
    (a) is absent from the base checkpoint (a new finetune param, e.g. the tactile encoder /
    dream head), (b) matches ``skip_regex`` (the resized action head), or (c) has a different
    shape than the base. Otherwise it is copied from the base (dtype-cast to the reference).
    The result therefore has EXACTLY the fresh key set, so ``check_pytree_equality`` passes and
    the fresh entries are subsequently left at init by the training loop.
    """

    params_path: str
    skip_regex: str = r"(.*/)?(action_in_proj|action_out_proj|state_proj)/.*"

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        pattern = re.compile(self.skip_regex)
        result = {}
        reinit, new = [], []
        for k, ref_v in flat_ref.items():
            loaded_v = flat_loaded.get(k)
            shape_mismatch = loaded_v is not None and getattr(loaded_v, "shape", None) != getattr(ref_v, "shape", None)
            if loaded_v is None:
                # New param not present in the base checkpoint (tactile encoder/teacher, dream head).
                result[k] = ref_v
                new.append(k)
            elif pattern.fullmatch(k) or shape_mismatch:
                # Resized / incompatible param -> leave at fresh init.
                result[k] = ref_v
                reinit.append(k)
            else:
                result[k] = loaded_v.astype(ref_v.dtype) if loaded_v.dtype != ref_v.dtype else loaded_v
        if reinit:
            logger.info("PartialCheckpointWeightLoader: reinitializing %d resized param(s): %s", len(reinit), sorted(reinit))
        if new:
            logger.info("PartialCheckpointWeightLoader: %d new param(s) from fresh init (not in base): %s", len(new), sorted(new)[:8])
        return flax.traverse_util.unflatten_dict(result, sep="/")


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
