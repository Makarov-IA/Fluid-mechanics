from pathlib import Path
import sys,copy,shutil,json,hashlib
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
import numpy as np,yaml
from solver_python.config import load_config
from solver_python.solver import Solver
from solver_python.snapshot_io import read_snapshot
from solver_python.projection import full_state_basis
from solver_python.stability import SteadyProblem,schur_operator,apply
src=root/'results/pulsation_re260';dst=root/'results/pulsation_re260_base'
assert not dst.exists(), 'Comparison base already exists'
base=load_config(root/'configs/pulsation_re260/pulsation_re260.yaml')
# Confirm that the unchanged baseline pipeline actually reproduces saved fields.
s=Solver(base)
for _ in range(1001):s.step()
a=read_snapshot(src/'binaries/result_1000.bin')
np.testing.assert_array_equal(s.psi,a.psi);np.testing.assert_array_equal(s.omega,a.omega)
with np.load(src/'binaries/forcing.npz') as old:
 np.testing.assert_array_equal(old['source'],s.source)
for section in ['binaries','newton','stationary']:
 shutil.copytree(src/section,dst/section)
mode_dir=dst/'linear_stability';mode_dir.mkdir()
with np.load(src/'linear_stability/stabilization.npz') as data:
 bundle={k:data[k] for k in data.files}
bundle['state_basis']=full_state_basis(bundle['basis'],base.nx,base.ny,s.dx,s.dy)
bundle['source']=s.source
bundle['wall_velocities']=np.zeros((4,2))
np.savez(mode_dir/'stabilization.npz',**bundle)
problem=SteadyProblem(base);z=problem.pack(bundle['psi_star'],bundle['omega_star'])
res=problem.solver_residuals(z);assert max(res)<1e-10
parts=schur_operator(problem,problem.jacobian(z));q=bundle['basis'].T
aq=np.column_stack([apply(parts,q[:,i]) for i in range(q.shape[1])]);small=q.T@aq
np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(small)),np.sort_complex(bundle['eigenvalues']),atol=1e-8,rtol=1e-8)
shutil.copy2(src/'linear_stability/eigenvalues.csv',mode_dir/'eigenvalues.csv')
report={'baseline_reused_from':str(src),'baseline_steps_replayed_bitwise':1001,
        'baseline_total_steps':1000000,'equilibrium_residuals':list(res),
        'unstable_eigenvalues':[[v.real,v.imag] for v in bundle['eigenvalues']],
        'state_basis_rank':len(bundle['state_basis']),
        'state_metric':'unweighted Euclidean inner product of concatenated interior psi and omega',
        'spectrum_scope':'physical operator with psi eliminated, not full-state feedback spectrum',
        'basis_file_sha256':hashlib.sha256((mode_dir/'stabilization.npz').read_bytes()).hexdigest()}
(mode_dir/'report.json').write_text(json.dumps(report,indent=2))
summary=json.loads((dst/'binaries/summary.json').read_text());summary.update(experiment='pulsation_re260_base',output_dir=str(dst/'binaries'),reused_from=str(src/'binaries'))
(dst/'binaries/summary.json').write_text(json.dumps(summary,indent=2))
for exp,method in [('pulsation_re260_base',None),('pulsation_re260_projection','projection'),('pulsation_re260_rhs','rhs')]:
 data=copy.deepcopy(base.resolved)
 data['forcing']['file']='../../solver_python/sources/harmonics.py'
 if method:
  data['stabilization']={'file':'../../results/pulsation_re260_base/linear_stability/stabilization.npz','gain':1.,'method':method,'components':'state'}
 directory=root/'configs'/exp;directory.mkdir()
 (directory/f'{exp}.yaml').write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True))
 text='Базовый расчёт без коррекции; проверенные результаты переиспользованы из pulsation_re260.' if method is None else ('Полная проекция пары (ψ,ω) после каждого обычного шага; следующее состояние исправлено.' if method=='projection' else 'Поправка −dt·P(z−z*) в обеих правых частях; коэффициенты заморожены в начале шага, gain=1.')
 (directory/'README.md').write_text(f'# {exp}\n\n{text}\n\nОбщие параметры: область 1×0.5, 201×101, Re=260, dt=1e-5, 10⁶ шагов до псевдовремени 10, нулевое начальное состояние и неподвижные стенки. Аракава, прежняя факторизация.\n\nПроекция по полной паре внутренних ψ и ω, обычная евклидова норма без весов; общий коэффициент для обеих компонент. Файл базиса один для двух способов. Кандидат и Ньютон найдены по базовой задаче; моды построены по физическому оператору, а не по карте шага псевдовремени.\n\nПолное вычитание и RHS с gain=1 имеют разную силу воздействия на один шаг: это часть сравнения, не доказательство преимущества порядка операций при равных усилениях.\n')
print(json.dumps(report,indent=2))
