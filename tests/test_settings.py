import json

import pytest

from airmouse.config import Config
from airmouse.settings import PARAMS, apply_values, load_config, save_config, schema, to_dict


def test_every_param_exists_on_config_with_default_in_range():
    cfg = Config()
    for p in PARAMS:
        value = getattr(cfg, p.name)
        assert p.min <= value <= p.max, p.name


def test_apply_clamps_and_rounds_integers():
    cfg = Config()
    changed = apply_values(cfg, {"cursor_speed": 99.0, "push_threshold": -5,
                                 "scroll_speed": 1234.6})
    assert cfg.cursor_speed == 5.0
    assert cfg.push_threshold == 0.1
    assert cfg.scroll_speed == 1235 and isinstance(cfg.scroll_speed, int)
    assert changed == {"cursor_speed": 5.0, "push_threshold": 0.1, "scroll_speed": 1235}


def test_apply_reports_only_real_changes():
    cfg = Config()
    assert apply_values(cfg, {"push_threshold": cfg.push_threshold}) == {}


@pytest.mark.parametrize("values", [
    {"no_such_setting": 1},
    {"push_threshold": "0.5"},
    {"push_threshold": True},
    {"push_threshold": None},
    {"smoothing_alpha": float("nan")},
    {"cursor_speed": float("inf")},
])
def test_apply_rejects_bad_input(values):
    with pytest.raises(ValueError):
        apply_values(Config(), values)


def test_apply_is_all_or_nothing():
    cfg = Config()
    with pytest.raises(ValueError):
        apply_values(cfg, {"push_threshold": 0.5, "bogus": 1})
    assert cfg.push_threshold == Config().push_threshold


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    cfg = Config()
    apply_values(cfg, {"smoothing_alpha": 0.4, "camera_index": 2})
    save_config(cfg, path)
    assert to_dict(load_config(path)) == to_dict(cfg)
    assert not path.with_suffix(".tmp").exists()


def test_missing_file_gives_defaults(tmp_path):
    assert to_dict(load_config(tmp_path / "nope.json")) == to_dict(Config())


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"push_threshold": "high"}',
                                     '{"smoothing_alpha": NaN}'])
def test_bad_file_is_set_aside_and_defaults_used(tmp_path, content):
    path = tmp_path / "settings.json"
    path.write_text(content)
    assert to_dict(load_config(path)) == to_dict(Config())
    assert path.with_suffix(".bad.json").exists() and not path.exists()


def test_unknown_keys_in_file_are_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"push_threshold": 0.5, "from_a_future_version": 1}))
    assert load_config(path).push_threshold == 0.5


def test_schema_is_json_serializable():
    json.dumps(schema())
