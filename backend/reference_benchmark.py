"""Independent Nashpy comparison with reproducible timing and Nash-gap checks.

Run: python reference_benchmark.py --output benchmark_reference.json
Nashpy is an optional *benchmark* dependency, never a runtime API dependency.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from hybrid_nash import solve_hybrid


def gap(a, b, x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != (a.shape[0],) or y.shape != (a.shape[1],):
        return float('inf')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or min(x.min(), y.min()) < -1e-7 or abs(x.sum()-1)>1e-7 or abs(y.sum()-1)>1e-7:
        return float('inf')
    return float(max(0, (a@y).max()-x@a@y, (x@b).max()-x@b@y))


def cases():
    rng = np.random.default_rng(20261103)
    yield 'matching_pennies', np.array([[1.,-1.],[-1.,1.]]), np.array([[-1.,1.],[1.,-1.]])
    yield 'prisoners_dilemma', np.array([[-1.,-3.],[0.,-2.]]), np.array([[-1.,0.],[-3.,-2.]])
    yield 'constant_2x2', np.ones((2,2)), np.ones((2,2))
    yield 'rectangular_2x3', np.array([[2.,0.,1.],[0.,2.,1.]]), np.array([[0.,2.,1.],[2.,0.,1.]])
    for n in (2,3,4,5):
        for i in range(3):
            yield f'random_{n}x{n}_{i}', rng.normal(size=(n,n)), rng.normal(size=(n,n))


def timed(fn):
    start=time.perf_counter_ns()
    result=fn()
    return result,(time.perf_counter_ns()-start)/1e6


def run(max_pairs=100000):
    import nashpy as nash
    rows=[]
    for name,a,b in cases():
        hybrid,ms=timed(lambda:solve_hybrid(a,b,max_pairs=max_pairs))
        ref,ref_ms=timed(lambda:list(nash.Game(a,b).support_enumeration(non_degenerate=False)))
        ours=[gap(a,b,e['row_probabilities'],e['col_probabilities']) for e in hybrid['equilibria']]
        refs=[gap(a,b,x,y) for x,y in ref]
        row={'case':name,'shape':list(a.shape),'hybrid_ms':ms,'reference_ms':ref_ms,
             'hybrid_count':len(ours),'reference_count':len(refs),
             'hybrid_best_gap':min(ours,default=None),'reference_best_gap':min(refs,default=None),
             'hybrid_method':hybrid['method'],'hybrid_complete':hybrid['complete_support_coverage'],
             'reference_valid_count':sum(g<=1e-6 for g in refs),
             'hybrid_valid_count':sum(g<=1e-6 for g in ours)}
        rows.append(row)
    return {'seed':20261103,'reference':'nashpy.support_enumeration(non_degenerate=False)',
            'notes':'Timings are single-process measurements; Nashpy support enumeration is not exhaustive for every degenerate equilibrium continuum. Counts are not necessarily comparable.',
            'cases':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='benchmark_reference.json')
    args=p.parse_args()
    result=run()
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'cases':len(result['cases']),'all_hybrid_valid':all(r['hybrid_valid_count']==r['hybrid_count'] for r in result['cases']),'output':args.output}))
