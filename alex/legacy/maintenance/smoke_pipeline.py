from pathlib import Path
import sys,copy,json,subprocess
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
import yaml,numpy as np
from solver_python.config import load_config
from solver_python.run import run
from solver_python.render import render
from solver_python.verify_magnets import verify
from solver_python.stability import SteadyProblem,newton,check_jacobian,schur_operator,apply
from solver_python.snapshot_io import read_snapshot
out=root/'legacy/maintenance';reports={}
for exp in ['pulsation_re260','pulsation_re260_stabilized','magnets_2x3_re260']:
 cfg=load_config(root/f'configs/{exp}/{exp}.yaml')
 data=copy.deepcopy(cfg.resolved)
 data['time'].update(mode='fixed_dt_steps',n_time_steps=101)
 data['output'].update(save_every_step=50,log_every_step=50)
 path=out/f'maintenance_{exp}.yaml';path.write_text(yaml.safe_dump(data,sort_keys=False))
 smoke=load_config(path)
 reports[exp]=run(smoke)
 if exp=='pulsation_re260':
  render(path,20,True)
  d=smoke.result_dir/'stationary'
  subprocess.run([sys.executable,str(root/'solver_python/find_stationary.py'),
   '--results-dir',str(smoke.output_dir),'--config',str(path),
   '--metrics-csv',str(d/'differences.csv'),'--plot-png',str(d/'differences.png'),
   '--stationary-snapshot',str(d/'state.bin'),'--streamplot-png',str(d/'state.png'),
   '--skip-newest','0'],check=True)
 if exp=='magnets_2x3_re260':
  reports['magnetic_verification']=verify(path,101,smoke.result_dir/'verification')
# Validate the unchanged equilibrium and unstable invariant subspace at full resolution.
cfg=load_config(root/'configs/pulsation_re260/pulsation_re260.yaml');problem=SteadyProblem(cfg)
s=read_snapshot(cfg.result_dir/'newton/equilibrium.bin');z=problem.pack(s.psi,s.omega)
z,history=newton(problem,z,cfg.steady_tolerance)
jac=problem.jacobian(z);jcheck=check_jacobian(problem,z,jac,np.random.default_rng(0))
with np.load(cfg.result_dir/'linear_stability/stabilization.npz') as data:
 q=data['basis'].T
 aq=np.column_stack([apply(schur_operator(problem,jac),q[:,i]) for i in range(q.shape[1])])
 small=q.T@aq;error=float(np.max(np.abs(aq-q@small)))
 assert error<1e-8
 np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(small)),np.sort_complex(data['eigenvalues']),rtol=1e-8,atol=1e-8)
reports['equilibrium']={'max_residual':history[-1]['max_residual'],'jacobian_errors':jcheck,'invariance_error':error}
(out/'pipeline_verification.json').write_text(json.dumps(reports,indent=2))
print('PIPELINE VERIFIED',reports['equilibrium'],flush=True)
