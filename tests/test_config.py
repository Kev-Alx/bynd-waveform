import json

import pytest
import yaml

from beyondwaveform.config import AttackConfig, load_config


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_config_preserves_scientific_numbers_and_unicode(tmp_path, extension):
    config = {"label": "語音 — café", "attack": {"eps": 1e-8, "learning_rate": 0.02},
              "enabled": True, "optional": None}
    path = tmp_path / ("config." + extension)
    path.write_text(json.dumps(config, ensure_ascii=False) if extension == "json"
                    else yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    loaded = load_config(path)
    assert loaded == config
    assert isinstance(loaded["attack"]["eps"], float)
    assert AttackConfig(**loaded["attack"]).eps == 1e-8


def test_json_config_requires_a_mapping(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration mapping"):
        load_config(path)
