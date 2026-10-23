from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'solver_python/config.py';s=p.read_text()
s=s.replace('class ConfigError(ValueError):','ALEX = Path(__file__).resolve().parents[1]\nRESULTS_ROOT = ALEX / "results"\nDEFAULT_CONFIG = ALEX / "configs/pulsation_re260/pulsation_re260.yaml"\n\n\nclass ConfigError(ValueError):')
s=s.replace('    stabilization: dict | None = None','    stabilization: dict | None = None\n\n    @property\n    def result_dir(self) -> Path:\n        return self.output_dir.parent')
s=s.replace('{"version", "experiment", "grid", "physics", "time",','{"version", "grid", "physics", "time",')
s=s.replace('experiment = raw.get("experiment")','experiment = path.stem')
s=s.replace('experiment must be a nonempty directory name using letters/digits/_.-','YAML filename must use letters/digits/_.-')
start=s.index('    output = section(');stop=s.index('    save_every = ',start)
s=s[:start]+'''    output = section(raw.get("output", {}), "output", {"save_every_step", "log_every_step"})
    out_dir = RESULTS_ROOT / experiment / "binaries"
'''+s[stop:]
s=s.replace('"version": 1, "experiment": experiment,','"version": 1,')
s=s.replace('"output": {"root": str(root.resolve()), "save_every_step": save_every, "log_every_step": log_every},','"output": {"save_every_step": save_every, "log_every_step": log_every},')
p.write_text(s)
for p in (root/'solver_python').rglob('*.py'):
 s=p.read_text().replace('from common.snapshot_io import','from solver_python.snapshot_io import')
 if 'Path(__file__).with_name("config.yaml")' in s:
  s=s.replace('Path(__file__).with_name("config.yaml")','DEFAULT_CONFIG')
  s=s.replace('from solver_python.config import ', 'from solver_python.config import DEFAULT_CONFIG, ',1)
 s=s.replace('root = Path(cfg.resolved["output"]["root"])','root = cfg.result_dir.parent')
 s=s.replace('root/"figures"/cfg.experiment','cfg.result_dir/"figures"')
 s=s.replace('root/"videos"/cfg.experiment','cfg.result_dir/"videos"')
 s=s.replace('root/"stationary"/cfg.experiment','cfg.result_dir/"stationary"')
 s=s.replace('root/"newton"/cfg.experiment','cfg.result_dir/"newton"')
 s=s.replace('root/"linear_stability"/cfg.experiment','cfg.result_dir/"linear_stability"')
 s=s.replace('root/"binaries"/uncontrolled','root/uncontrolled/"binaries"')
 s=s.replace('root/"binaries"/args.uncontrolled','root/args.uncontrolled/"binaries"')
 s=s.replace('root/"tables"/experiment','root/experiment/"tables"')
 s=s.replace('Path(cfg.resolved["output"]["root"])/"figures"/cfg.experiment','cfg.result_dir/"figures"')
 s=s.replace('ALEX/"scripts/python/', 'ALEX/"solver_python/')
 s=s.replace('The working YAML','The selected YAML').replace('the working YAML','the selected YAML').replace('ONE working YAML','selected YAML')
 p.write_text(s)
for name in ['plot_fields.py','find_stationary.py']:
 p=root/'solver_python'/name;s=p.read_text()
 s=s.replace('ALEX_DIR = SCRIPT_DIR.parents[1]\nsys.path.insert(0, str(ALEX_DIR / "common"))\n\nimport snapshot_io','ALEX_DIR = SCRIPT_DIR.parent\nsys.path.insert(0, str(ALEX_DIR))\n\nfrom solver_python import snapshot_io')
 if name=='find_stationary.py':
  start=s.index('def parse_config(');stop=s.index('\ndef collect_result_files(',start)
  s=s[:start]+'''def parse_config(path: Path) -> dict:
    from solver_python.config import load_config
    cfg = load_config(path)
    return {"dt": cfg.dt, "n_time_steps": cfg.n_time_steps}

'''+s[stop:]
 p.write_text(s)
# Tests use a stable baseline config and isolate all generated data under pytest's tmp_path.
for p in (root/'solver_python/tests').glob('test_*.py'):
 s=p.read_text()
 import re
 s=re.sub(r'CONFIG = Path\(__file__\)\.parents\[1\]\s*/\s*"config.yaml"','CONFIG = Path(__file__).parents[2] / "configs/pulsation_re260/pulsation_re260.yaml"\nSOURCE = Path(__file__).parents[1] / "sources/harmonics.py"',s)
 s=s.replace('CONFIG.parent/"sources/harmonics.py"','SOURCE')
 s=s.replace('root=str(tmp_path), ','').replace('root=str(tmp_path),','')
 s=s.replace('"version": 1, "experiment": "cpp_comparison",','"version": 1,')
 s=s.replace('raw["experiment"] = "limit"','')
 s=s.replace('root/"cpu/solver/solver.cpp"','root/"legacy/cpu/solver/solver.cpp"')
 s=s.replace('{root}/cpu','{root}/legacy/cpu')
 s=s.replace('root/"cpu/forcing/omega_forcing.cpp"','root/"legacy/cpu/forcing/omega_forcing.cpp"')
 if p.name=='test_config.py':
  s=s.replace('    data["output"]["root"] = "relative"\n','')
  s=s.replace('assert cfg.output_dir == tmp_path/"relative"/"binaries"/cfg.experiment','assert cfg.output_dir == tmp_path/"results"/"config"/"binaries"\n    assert cfg.experiment == path.stem')
 p.write_text(s)
(root/'solver_python/tests/conftest.py').write_text('''import pytest
from solver_python import config

@pytest.fixture(autouse=True)
def isolated_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_ROOT", tmp_path / "results")
''')
