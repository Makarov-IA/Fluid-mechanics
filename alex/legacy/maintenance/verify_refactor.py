from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from solver_python.config import load_config
from solver_python.solver import Solver
root=Path(__file__).resolve().parents[2];out=root/'legacy/maintenance';report={}
for name,exp in [('baseline','pulsation_re260'),('controlled','pulsation_re260_stabilized')]:
 cfg=load_config(root/f'configs/{exp}/{exp}.yaml')
 s=Solver(cfg);s.warm_up();history=[]
 for i in range(5000):history.append(s.step())
 with np.load(out/f'{name}_reference.npz') as ref:
  for field,value in [('psi',s.psi),('omega',s.omega),('residuals',np.asarray(history))]:
   np.testing.assert_array_equal(value,ref[field])
 timings=[]
 for repeat in range(5):
  start=time.perf_counter()
  for i in range(2000):s.step()
  timings.append((time.perf_counter()-start)/2000*1e3)
 report[name]={'bitwise_equal_steps':5000,'ms_per_step':timings,'median':float(np.median(timings))}
 print(name,report[name],flush=True)
(out/'after_benchmark.json').write_text(json.dumps(report,indent=2))
