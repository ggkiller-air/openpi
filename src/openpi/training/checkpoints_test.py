import json

import pytest

from openpi.models import pi0_config
from openpi.training import checkpoints


def test_jepa_model_config_round_trip(tmp_path):
    baseline = pi0_config.Pi0Config(pi05=True, discrete_state_input=False)
    dream = pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        use_tactile=True,
        use_tactile_dream=True,
        tactile_encoder_type="coord",
        dream_horizon=3,
        tactile_dream_beta=0.75,
        dream_state=True,
        dream_vision=True,
        vision_horizon=2,
    )

    checkpoints.save_jepa_model_config(tmp_path, dream)
    restored = checkpoints.load_jepa_model_config(tmp_path, baseline)

    assert restored.use_tactile
    assert restored.use_tactile_dream
    assert restored.tactile_encoder_type == "coord"
    assert restored.dream_horizon == 3
    assert restored.tactile_dream_beta == 0.75
    assert restored.dream_state
    assert restored.dream_vision
    assert restored.vision_horizon == 2


def test_jepa_model_config_rejects_unknown_field(tmp_path):
    (tmp_path / "jepa_model_config.json").write_text(json.dumps({"unknown": True}))

    with pytest.raises(ValueError, match="unknown JEPA model config fields"):
        checkpoints.load_jepa_model_config(tmp_path, pi0_config.Pi0Config())
