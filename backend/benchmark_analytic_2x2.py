"""Seeded 2x2 benchmark with validity and latency distribution reporting."""
import json, time, statistics
import numpy as np
from analytic_2x2 import solve_2x2
from lemke_howson import solve_lemke_howson

def run(seed=20261103, cases=100):
    rng=np.random.default_rng(seed)
    results={name:{'latency_ms':[],'failures':0} for name in ('analytic','nashpy_lh')}
    for _ in range(cases):
        a=rng.integers(-5,6,(2,2)).astype(float)
        b=rng.integers(-5,6,(2,2)).astype(float)
        for name,fn in [('analytic',lambda:solve_2x2(a,b)),('nashpy_lh',lambda:solve_lemke_howson(a,b))]:
            start=time.perf_counter()
            try:
                r=fn()
                if not r['equilibria'] or any(e['nash_gap']>1e-7 for e in r['equilibria']):
                    results[name]['failures']+=1
            except Exception:
                results[name]['failures']+=1
            results[name]['latency_ms'].append((time.perf_counter()-start)*1000)
    for r in results.values():
        vals=r.pop('latency_ms')
        r.update(median_ms=statistics.median(vals),p90_ms=float(np.percentile(vals,90)),mean_ms=statistics.mean(vals),min_ms=min(vals),max_ms=max(vals))
    return {'seed':seed,'cases':cases,'results':results}

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
