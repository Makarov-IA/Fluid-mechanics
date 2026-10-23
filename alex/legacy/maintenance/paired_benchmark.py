from pathlib import Path
import sys,importlib.util,time,json
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
import numpy as np
from solver_python.config import load_config
from solver_python.solver import Solver
folder=root/'legacy/maintenance'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
oldk=module('solver_python._oldkernels',folder/'before_cleanup/solver_python/kernels.py')
olds=module('solver_python._oldsolver',folder/'before_cleanup/solver_python/solver.py');olds.k=oldk
report={}
for name in ['pulsation_re260','pulsation_re260_stabilized']:
 cfg=load_config(root/f'configs/{name}/{name}.yaml')
 a,b=olds.Solver(cfg),Solver(cfg)
 for s in (a,b):s.warm_up()
 times={'before':[],'after':[]}
 for repeat in range(6):
  for label,s in ([('before',a),('after',b)] if repeat%2==0 else [('after',b),('before',a)]):
   start=time.perf_counter()
   for i in range(3000):s.step()
   times[label].append((time.perf_counter()-start)/3000*1000)
 np.testing.assert_array_equal(a.psi,b.psi);np.testing.assert_array_equal(a.omega,b.omega)
 np.testing.assert_array_equal(a.last_residuals,b.last_residuals)
 report[name]={**times,'median_before':float(np.median(times['before'])),'median_after':float(np.median(times['after'])),'bitwise_equal_steps':18000}
 print(name,report[name],flush=True)
(folder/'paired_benchmark.json').write_text(json.dumps(report,indent=2))
