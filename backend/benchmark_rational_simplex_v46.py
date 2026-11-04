"""Reproducible exact-support feasibility benchmark, not a SOTA claim.

Execute from backend/: python benchmark_rational_simplex_v46.py
Measures equivalent feasibility tasks with both exact-rational backends.
"""
import json
import statistics
import time
from fractions import Fraction
import numpy as np
from exact_support import _feasible_strategy_by_vertices
from exact_rational_lp import feasible_strict_strategy


def run(repetitions=3):
    rng=np.random.default_rng(2026)
    results=[]
    for size in (4,5,6,7,8):
        payoff=[[Fraction(0)]*size]
        payoff += [[Fraction(int(z)) for z in rng.integers(-9,0,size=size)]
                   for _ in range(size-1)]
        support=tuple(range(size))
        methods=(('rational_simplex',feasible_strict_strategy),
                 ('vertex_enumeration',_feasible_strategy_by_vertices))
        entries={}
        for name,solver in methods:
            times=[];last_work=0
            for _ in range(repetitions):
                work={'steps':0,'limit':2000000}
                start=time.perf_counter()
                witness=solver(payoff,(0,),support,work)
                times.append(time.perf_counter()-start)
                assert witness is not None and sum(witness)==1 and all(v>0 for v in witness)
                assert all(sum(v*z for v,z in zip(row,witness))<0 for row in payoff[1:])
                last_work=work['steps']
            entries[name]={'median_seconds':statistics.median(times),
                           'reported_work_units':last_work}
        results.append({'actions':size,'solvers':entries})
    return {'seed':2026,'repetitions':repetitions,'case':'one dominant action and all opponent actions supported',
            'note':'Wallclock timings depend on hardware; work-unit definitions differ between backends.',
            'results':results}


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
