"""Reproducible local benchmark; run: python benchmark_hybrid.py"""
import json, time, statistics
import numpy as np
from hybrid_nash import solve_hybrid
from support_lp import solve_support_lp


def run(repeats=3):
    rng=np.random.default_rng(20261103)
    rows=[]
    for n in (2,3,4):
        for kind in ('random','constant'):
            a=rng.normal(size=(n,n)) if kind=='random' else np.zeros((n,n))
            b=rng.normal(size=(n,n)) if kind=='random' else np.zeros((n,n))
            for method,fn in [('hybrid',lambda:solve_hybrid(a,b)),('lp',lambda:solve_support_lp(a,b))]:
                samples=[]; result=None
                for _ in range(repeats):
                    t=time.perf_counter_ns();result=fn();samples.append((time.perf_counter_ns()-t)/1e6)
                rows.append({'size':n,'kind':kind,'method':method,'median_ms':round(statistics.median(samples),4),
                             'equilibria':len(result['equilibria']), 'max_gap':max((e['nash_gap'] for e in result['equilibria']),default=None)})
    return rows

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
