"""Data transforms for the Unitree G1 SONIC embodiment (GR00T `unitree_g1_sonic`).

This pipes the GR00T SONIC LeRobot dataset (e.g. ``carry-bucket-stereo``) into a pi0.5
model and back. The model is finetuned to output the SONIC action space:

    motion_token (64, SONIC latent) | left_hand_joints (7) | right_hand_joints (7)  -> 78-d

at an action horizon of 40. State is the 46-d vector defined by the registered
``unitree_g1_sonic`` modality config (this is exactly what the bridge sends at inference):

    left_leg(6) right_leg(6) waist(3) left_arm(7) right_arm(7) left_hand(7) right_hand(7) projected_gravity(3)


"""

from collections.abc import Mapping
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
SONIC_ACTION_HORIZON = 40
SONIC_PROTOCOL = "sonic_vla_v1"
SONIC_VIDEO_KEYS = ("ego_view_left", "ego_view_right")
SONIC_TACTILE_KEYS = ("vest", "left_arm", "right_arm")
SONIC_TACTILE_DIM = 768


def make_sonic_metadata(model_config) -> dict:
    """Describe the websocket contract restored from a SONIC checkpoint."""
    action_dim = int(model_config.action_dim)
    action_horizon = int(model_config.action_horizon)
    if action_dim != SONIC_ACTION_DIM or action_horizon != SONIC_ACTION_HORIZON:
        raise ValueError(
            "SONIC checkpoint must use action_dim=78 and action_horizon=40; "
            f"got {action_dim} and {action_horizon}"
        )
    return {
        "protocol": SONIC_PROTOCOL,
        "state_dim": SONIC_STATE_DIM,
        "action_horizon": SONIC_ACTION_HORIZON,
        "action_dim": SONIC_ACTION_DIM,
        "video_keys": list(SONIC_VIDEO_KEYS),
        "requires_tactile": bool(getattr(model_config, "use_tactile", False)),
        "action_layout": {
            "motion_token": [0, MOTION_TOKEN_DIM],
            "left_hand_joints": [MOTION_TOKEN_DIM, MOTION_TOKEN_DIM + LEFT_HAND_DIM],
            "right_hand_joints": [MOTION_TOKEN_DIM + LEFT_HAND_DIM, SONIC_ACTION_DIM],
        },
    }

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
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.ndim == 4 and image.shape[1] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "t c h w -> t h w c")
    if image.ndim not in (3, 4) or image.shape[-1] != 3:
        raise ValueError(f"expected HWC/CHW or THWC/TCHW RGB image, got {image.shape}")
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
    parts = [s[..., spans[g][0] : spans[g][1]] for g in STATE_GROUP_ORDER]
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
    dream_state: bool = False
    dream_vision: bool = False
    vision_horizon: int = 4
    requires_tactile: bool = False

    def __call__(self, data: dict) -> dict:
        if "state" in data:
            # Inference path: bridge already sends the canonical 46-d vector.
            state = np.asarray(data["state"], dtype=np.float32)
        else:
            # Training path: reassemble from the raw 43-d column + projected_gravity,
            # using spans derived from the dataset's modality.json (train == inference order).
            state = assemble_state_46(data["state_43"], data["projected_gravity"], self.state_spans)
        if state.shape[-1] != SONIC_STATE_DIM:
            raise ValueError(
                f"SONIC state must have width {SONIC_STATE_DIM}, got {state.shape}"
            )
        if not np.isfinite(state).all():
            raise ValueError("SONIC state contains NaN or infinity")
        if self.dream_state:
            if state.ndim == 1:
                state = state[None, :]
            elif state.ndim != 2:
                raise ValueError(f"state-JEPA expects [D] or [T, D], got {state.shape}")

        # Head stereo -> two of pi's three fixed image slots; the third is masked.
        # Slot names are opaque to the model (shared vision encoder); only train/inference
        # consistency matters. ego_view_left -> base_0_rgb, ego_view_right -> left_wrist_0_rgb.
        left_image = _parse_image(data["ego_view_left"])
        right_image = _parse_image(data["ego_view_right"])
        left_current = left_image[0] if left_image.ndim == 4 else left_image
        right_current = right_image[0] if right_image.ndim == 4 else right_image

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": left_current,
                "left_wrist_0_rgb": right_current,
                "right_wrist_0_rgb": np.zeros_like(left_current),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Only mask padding images for pi0/pi0.5, not pi0-FAST.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # A sequence is supplied only by the training loader. Inference remains compatible
        # with one current frame and does not need to fabricate future JEPA targets.
        if self.dream_vision and left_image.ndim == 4 and right_image.ndim == 4:
            expected = self.vision_horizon + 1
            if left_image.shape[0] != expected or right_image.shape[0] != expected:
                raise ValueError(
                    f"vision-JEPA expects {expected} stereo frames, got {left_image.shape[0]} and "
                    f"{right_image.shape[0]}"
                )
            inputs["future_images"] = {
                "base_0_rgb": left_image[1:],
                "left_wrist_0_rgb": right_image[1:],
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

        # Training supplies three windowed device streams; inference supplies their
        # already-concatenated 768-wide current frame.
        if self.requires_tactile and "tactile" not in data:
            raise ValueError("This SONIC checkpoint requires a current tactile frame")
        if "tactile" in data:
            source = data["tactile"]
            if isinstance(source, Mapping):
                missing = [key for key in SONIC_TACTILE_KEYS if key not in source]
                if missing:
                    raise ValueError(f"SONIC tactile is missing device streams: {missing}")
                t = np.concatenate([np.asarray(source[key]) for key in SONIC_TACTILE_KEYS], axis=-1)
            else:
                t = np.asarray(source)
            if t.dtype != np.uint8:
                # LeRobot materializes Arrow list<uint8> values as int64 arrays. Preserve
                # the wire-level uint8 contract while accepting that lossless loader cast.
                if not np.issubdtype(t.dtype, np.integer) or (
                    t.size and (t.min() < 0 or t.max() > 255)
                ):
                    raise ValueError(
                        f"SONIC tactile must contain uint8-compatible integers, got {t.dtype}"
                    )
                t = t.astype(np.uint8)
            if t.ndim == 1:
                t = t[None, :]
            if t.ndim != 2 or t.shape[-1] != SONIC_TACTILE_DIM:
                raise ValueError(
                    f"SONIC tactile must have shape [{SONIC_TACTILE_DIM}] or "
                    f"[T, {SONIC_TACTILE_DIM}], got {t.shape}"
                )
            inputs["tactile"] = t

        return inputs


@dataclasses.dataclass(frozen=True)
class SonicOutputs(transforms.DataTransformFn):
    """Slice the model output back to the 78-d SONIC action (drop any padding).

    The GR00T-side bridge further splits this into motion_token[:64] /
    left_hand_joints[64:71] / right_hand_joints[71:78].
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :SONIC_ACTION_DIM], dtype=np.float32)
        if actions.shape != (SONIC_ACTION_HORIZON, SONIC_ACTION_DIM):
            raise ValueError(
                f"SONIC actions must have shape ({SONIC_ACTION_HORIZON}, "
                f"{SONIC_ACTION_DIM}), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("SONIC actions contain NaN or infinity")
        return {"actions": actions}
