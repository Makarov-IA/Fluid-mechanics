import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from solver_python.config import ConfigError, load_config, parse_config
from solver_python.forcing import build_forcing, dipole_field


CONFIG = Path(__file__).parents[2] / "configs/pulsation_re260/pulsation_re260.yaml"
SOURCE = Path(__file__).parents[1] / "sources/harmonics.py"


def raw():
    data = copy.deepcopy(load_config(CONFIG).resolved)
    data["physics"] = {"re": 260.}
    data["forcing"] = {"type": "custom", "file": str(SOURCE)}
    return data


def test_safe_yaml_duplicate_keys_and_tags(tmp_path):
    path = tmp_path/"config.yaml"
    path.write_text("version: 1\nversion: 2\n")
    with pytest.raises(ConfigError, match="Duplicate"):
        load_config(path)
    path.write_text("!!python/object/apply:builtins.print ['must not execute']")
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("group,key,value", [
    ("grid", "nx", 2), ("grid", "nx", True), ("grid", "lx", 0),
    ("physics", "re", 0), ("physics", "re", -1), ("physics", "re", "nan"),
    ("time", "dt", float("inf")), ("time", "dt", True), ("time", "n_time_steps", 1.5),
    ("execution", "threads", 0),
    ("output", "save_every_step", 0), ("grid", "typo", 100),
])
def test_invalid_values_are_rejected(tmp_path, group, key, value):
    data = raw()
    data[group][key] = value
    with pytest.raises(ConfigError):
        parse_config(data, tmp_path/"config.yaml")


def test_normal_wall_velocity_rejected(tmp_path):
    data = raw()
    data["boundaries"]["left"]["u"] = .1
    with pytest.raises(ConfigError, match="Normal"):
        parse_config(data, tmp_path/"config.yaml")


def test_yaml_scientific_notation_and_paths(tmp_path, monkeypatch):
    data = raw()
    data["time"]["dt"] = "1e-5"
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.chdir("/")
    cfg = load_config(path)
    assert cfg.dt == 1e-5
    assert cfg.output_dir == tmp_path/"results"/"config"/"binaries"
    assert cfg.experiment == path.stem


def test_fixed_steps_dt_and_ambiguous_time_rejected(tmp_path):
    data = raw()
    data["time"].update(mode="fixed_steps", t_max=1.)
    with pytest.raises(ConfigError):
        parse_config(data, tmp_path/"config.yaml")
    del data["time"]["dt"]
    cfg = parse_config(data, tmp_path/"config.yaml")
    assert cfg.dt == 1./cfg.n_time_steps


def dipole_config():
    data = raw()
    data["physics"]["re"] = 2400.
    data["forcing"] = {
        "type": "magnets", "magnets": [[.06, .03, .01, .005], [.18, .09, .01, -.005]],
        "density_kg_m3": 1000., "current_density_A_m2": [.03, .1], "plane_height_m": .002,
        "scales": {"length_m": .24, "velocity_m_s": .01, "kinematic_viscosity_m2_s": 1e-6}}
    return data


def test_dipole_units_and_re_consistency(tmp_path):
    data = dipole_config()
    cfg = parse_config(data, tmp_path/"config.yaml")
    xs, ys = np.linspace(0, 1, 17), np.linspace(0, .5, 9)
    source, bz = build_forcing(cfg, xs, ys)
    expected_b, dbx, dby = dipole_field(xs[:, None]*.24, ys[None, :]*.24,
                                      data["forcing"]["magnets"], .002)
    np.testing.assert_allclose(bz, expected_b)
    np.testing.assert_allclose(source, -(.03*dbx+.1*dby)/1000*(.24/.01)**2)
    data["physics"]["re"] = 260.
    with pytest.raises(ConfigError, match="disagrees"):
        parse_config(data, tmp_path/"config.yaml")


@pytest.mark.parametrize("magnet", [[0, 0, 0, 1], [0, 0, -.1, 1], [0, 0, .1], [0, 0, .1, "nan"]])
def test_invalid_dipoles_rejected(tmp_path, magnet):
    data = dipole_config()
    data["forcing"]["magnets"] = [magnet]
    with pytest.raises(ConfigError):
        parse_config(data, tmp_path/"config.yaml")


@pytest.mark.parametrize("body,error", [
    ("x = 1", "must define source"),
    ("def source(x,y,lx,ly): return float('nan')", "Non-finite"),
    ("def source(x,y,lx,ly): return [1, 2, 3]", "shape"),
    ("def source(x,y,lx,ly): return 1j", "real numeric"),
])
def test_invalid_custom_sources_fail(tmp_path, body, error):
    path = tmp_path/"custom.py"
    path.write_text(body)
    data = raw()
    data["forcing"] = {"type": "custom", "file": "custom.py"}
    cfg = parse_config(data, tmp_path/"config.yaml")
    with pytest.raises((ValueError, FloatingPointError), match=error):
        build_forcing(cfg, np.linspace(0,1,5), np.linspace(0,.5,4))


def test_custom_source_is_loaded_fresh_and_can_return_scalar(tmp_path):
    path = tmp_path/"custom.py"
    data = raw()
    data["forcing"] = {"type": "custom", "file": "custom.py"}
    for value in (2., -3.):
        path.write_text(f"def source(x,y,lx,ly): return {value}")
        cfg = parse_config(data, tmp_path/"config.yaml")
        source, _ = build_forcing(cfg, np.linspace(0,1,5), np.linspace(0,.5,4))
        np.testing.assert_array_equal(source, np.full((5,4),value))


def test_missing_custom_file_rejected(tmp_path):
    data = raw()
    data["forcing"] = {"type": "custom", "file": "missing.py"}
    with pytest.raises(ConfigError, match="does not exist"):
        parse_config(data, tmp_path/"config.yaml")


def test_experiment_name_is_yaml_stem(tmp_path):
    data = raw()
    cfg = parse_config(data, tmp_path/"another_experiment.yaml")
    assert cfg.experiment == "another_experiment"
    assert cfg.result_dir == tmp_path/"results"/"another_experiment"
    assert cfg.output_dir == cfg.result_dir/"binaries"
    data["experiment"] = "conflicting_name"
    with pytest.raises(ConfigError, match="Unknown root"):
        parse_config(data, tmp_path/"another_experiment.yaml")


def test_results_root_is_fixed_to_project(tmp_path):
    data = raw()
    data["output"]["root"] = "elsewhere"
    with pytest.raises(ConfigError, match="Unknown output"):
        parse_config(data, tmp_path/"experiment.yaml")
