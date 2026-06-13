"""Data transforms for the Unitree G1 SONIC embodiment (GR00T `unitree_g1_sonic`).

This pipes the GR00T SONIC LeRobot dataset (e.g. ``carry-bucket-stereo``) into a pi0.5
model and back. The model is finetuned to output the SONIC action space:

    motion_token (64, SONIC latent) | left_hand_joints (7) | right_hand_joints (7)  -> 78-d

at an action horizon of 40. State is the 46-d vector defined by the registered
``unitree_g1_sonic`` modality config (this is exactly what the bridge sends at inference):

    left_leg(6) right_leg(6) waist(3) left_arm(7) right_arm(7) left_hand(7) right_hand(7) projected_gravity(3)


"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Action layout produced/consumed by the model (concatenation order MUST match
# compute_norm_stats and the bridge's split).
MOTION_TOKEN_DIM = 64
LEFT_HAND_DIM = 7
RIGHT_HAND_DIM = 7
SONIC_ACTION_DIM = MOTION_TOKEN_DIM + LEFT_HAND_DIM + RIGHT_HAND_DIM  # 78

# 46-d state dim (8 groups of the registered unitree_g1_sonic modality config).
SONIC_STATE_DIM = 46

# Order in which the joint groups are concatenated into the state vector. This MUST equal
# the GR00T `unitree_g1_sonic` modality config's state `modality_keys` order (minus
# projected_gravity, which is a separate dataset column appended last) — i.e. exactly what
# the bridge sends at inference. Do not reorder.
STATE_GROUP_ORDER = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
)

# Span (start, end) of each joint group inside the raw 43-d `observation.state` column.
# These come from the dataset's meta/modality.json (carry-bucket-stereo shown here). The raw
# column orders left_hand BEFORE right_arm, which differs from STATE_GROUP_ORDER — that is the
# whole reason we slice-and-reorder. SonicDataConfig overrides these by reading the actual
# dataset's modality.json, so any SONIC dataset stays correct without editing this default.
DEFAULT_STATE_SPANS = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "left_hand": (22, 29),
    "right_arm": (29, 36),
    "right_hand": (36, 43),
}


def make_sonic_example() -> dict:
    """A random input example matching the *inference* contract (bridge side)."""
    return {
        "state": np.random.rand(SONIC_STATE_DIM).astype(np.float32),
        "ego_view_left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "ego_view_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "carry the bucket",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def assemble_state_46(state_43: np.ndarray, projected_gravity: np.ndarray, spans: dict | None = None) -> np.ndarray:
    """Reorder the raw 43-d state column into the 46-d inference-contract vector.

    Slices each joint group out of the raw column using ``spans`` (group -> (start, end),
    from the dataset's modality.json) and concatenates them in ``STATE_GROUP_ORDER`` — the
    same order the bridge sends at inference — then appends projected_gravity. Deriving the
    order this way guarantees train == inference regardless of how the raw column is laid out.
    """
    spans = spans or DEFAULT_STATE_SPANS
    s = np.asarray(state_43, dtype=np.float32)
    parts = [s[spans[g][0] : spans[g][1]] for g in STATE_GROUP_ORDER]
    parts.append(np.asarray(projected_gravity, dtype=np.float32))
    return np.concatenate(parts, axis=-1)


@dataclasses.dataclass(frozen=True)
class SonicInputs(transforms.DataTransformFn):
    """Map a SONIC dataset/inference sample into the pi0.5 model input format.

    Used for BOTH training and inference. Two state paths converge on the same 46-d vector:
      * inference (bridge): a pre-assembled ``state`` (46-d) is passed and used directly.
      * training (repack):  ``state_43`` + ``projected_gravity`` are reassembled by index.
    """

    # Determines which model will be used. Do not change for your own dataset.
    model_type: _model.ModelType

    # Span of each joint group in the raw 43-d observation.state column, read from the
    # dataset's modality.json by SonicDataConfig. Only used on the training path.
    state_spans: dict | None = None

    def __call__(self, data: dict) -> dict:
        if "state" in data:
            # Inference path: bridge already sends the canonical 46-d vector.
            state = np.asarray(data["state"], dtype=np.float32)
        else:
            # Training path: reassemble from the raw 43-d column + projected_gravity,
            # using spans derived from the dataset's modality.json (train == inference order).
            state = assemble_state_46(data["state_43"], data["projected_gravity"], self.state_spans)

        # Head stereo -> two of pi's three fixed image slots; the third is masked.
        # Slot names are opaque to the model (shared vision encoder); only train/inference
        # consistency matters. ego_view_left -> base_0_rgb, ego_view_right -> left_wrist_0_rgb.
        left_image = _parse_image(data["ego_view_left"])
        right_image = _parse_image(data["ego_view_right"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": left_image,
                "left_wrist_0_rgb": right_image,
                "right_wrist_0_rgb": np.zeros_like(left_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Only mask padding images for pi0/pi0.5, not pi0-FAST.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training. Concatenate the three SONIC action
        # columns into a single 78-d vector. ORDER MUST MATCH SonicOutputs / the bridge split.
        if "motion_token" in data:
            inputs["actions"] = np.concatenate(
                [
                    np.asarray(data["motion_token"], dtype=np.float32),
                    np.asarray(data["left_hand_joints"], dtype=np.float32),
                    np.asarray(data["right_hand_joints"], dtype=np.float32),
                ],
                axis=-1,
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Tactile passthrough (only present when use_tactile). Kept raw uint8 — NOT normalized;
        # the tactile encoder divides by 255 internally. Shape [T, 256]: training gives the
        # windowed [tactile_horizon, 256]; inference (bridge) gives a single [256] frame -> [1, 256].
        if "tactile" in data:
            t = np.asarray(data["tactile"])
            if t.ndim == 1:
                t = t[None, :]
            inputs["tactile"] = t.astype(np.uint8)

        return inputs


@dataclasses.dataclass(frozen=True)
class SonicOutputs(transforms.DataTransformFn):
    """Slice the model output back to the 78-d SONIC action (drop any padding).

    The GR00T-side bridge further splits this into motion_token[:64] /
    left_hand_joints[64:71] / right_hand_joints[71:78].
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :SONIC_ACTION_DIM])}
