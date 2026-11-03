"""Paired baseline vs solve-based pivot timings; run from backend directory."""
import importlib.util, pathlib, time, json, statistics
import numpy as np
import independent_lh as optimized
baseline_path = pathlib.Path(__file__).resolve().parent / 'independent_lh_v21_baseline.py'
spec = importlib.util.spec_from_file_location('baseline_lh', baseline_path)
baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
records=[]
for n in (2,3,5,8,10):
 for seed in range(5):
  rng=np.random.default_rng(10000+n*100+seed)
  a=rng.integers(-10,11,(n,n)).astype(float);b=rng.integers(-10,11,(n,n)).astype(float)
  for name,solver in [('baseline',baseline.solve_independent_lh),('optimized',optimized.solve_independent_lh)]:
   samples=[];valid=0
   for _ in range(3):
    t=time.perf_counter_ns();result=solver(a,b,max_labels=min(3,2*n));samples.append((time.perf_counter_ns()-t)/1e6)
    valid+=bool(result['equilibria'])
   records.append(dict(size=n,seed=seed,solver=name,median_ms=statistics.median(samples),valid_runs=valid,failures=len(result['failures'])))
summary={str(n):{name:round(statistics.median(r['median_ms'] for r in records if r['size']==n and r['solver']==name),4) for name in ('baseline','optimized')} for n in (2,3,5,8,10)}
out={'summary':summary,'records':records,'methodology':'5 seeded games per size; 3 repetitions; first 3 initial labels; no warmup; sequential baseline then optimized; exploratory'}
(pathlib.Path(__file__).parent/'benchmark_pivot_v22.json').write_text(json.dumps(out,indent=2))
print(json.dumps(summary,indent=2))
