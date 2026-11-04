"""Reproducible v48 mixed-dominance benchmark, with honest regression cases.

Run from backend/: python benchmark_mixed_dominance_v48.py
The three arms enumerate identical support prefixes with the same exact LP.
No favorable cases are filtered and no global performance claim is implied.
"""
import json
import platform
import statistics
import sys
import time
import numpy as np
import scipy
from general_exact_search import search_exact_nash_cells


def support_signature(result):
    return sorted((tuple(v['support']['row']),tuple(v['support']['col']))
                  for v in result['families'])


def run(repetitions=5, seed=48008):
    rng=np.random.default_rng(seed)
    cases=[]
    a=np.asarray([[0,0],[2,-1],[-1,2]],dtype=float)
    cases.append(('mixed_only_3x2',a,np.zeros_like(a)))
    a=np.asarray([[0,0,0,0],[2,-1,2,-1],[-1,2,-1,2],[0,0,0,0]],dtype=float)
    cases.append(('mixed_only_4x4',a,np.zeros_like(a)))
    cases.append(('all_ties_4x4',np.zeros((4,4)),np.zeros((4,4))))
    for n in (4,5,6):
        cases.append((f'seeded_random_{n}',rng.integers(-4,5,size=(n,n)),
                      rng.integers(-4,5,size=(n,n))))
    modes=(('no_pruning',False,False),('pure',True,False),
           ('pure_plus_mixed',True,True))
    outcomes=[]
    for label,a,b in cases:
        alternatives={}
        for mode,pure,mixed in modes:
            timings=[];returned=None
            for _ in range(repetitions):
                start=time.perf_counter()
                result=search_exact_nash_cells(
                    a,b,max_pairs=100,max_support=2,max_operations=1000000,
                    prune_supports=pure,mixed_pruning=mixed)
                timings.append(time.perf_counter()-start)
                if returned is not None and support_signature(result)!=support_signature(returned):
                    raise AssertionError('Nondeterministic support results')
                returned=result
            alternatives[mode]={
                'median_seconds':statistics.median(timings),
                'support_pairs_checked':returned['support_pairs_checked'],
                'complete_equilibrium_set':returned['complete_equilibrium_set'],
                'incomplete_reason':returned['incomplete_reason'],
                'feasible_support_pairs':len(returned['families']),
                'dominance_pruned_pairs':returned['dominance_pruned_pairs'],
                'mixed_dominance_pruned_pairs':returned['mixed_dominance_pruned_pairs'],
                'mixed_dominance_lp_attempts':returned['mixed_dominance_lp_attempts'],
                'rational_work_operations':returned['exact_operations'],
            }
            if mode=='no_pruning':
                baseline=returned
            elif (support_signature(result)!=support_signature(baseline) or
                  result['support_pairs_checked']!=baseline['support_pairs_checked'] or
                  result['complete_equilibrium_set']!=baseline['complete_equilibrium_set'] or
                  result['incomplete_reason']!=baseline['incomplete_reason']):
                raise AssertionError('Pruning changed exact equilibrium-set classification')
        outcomes.append({'case':label,'shape':list(a.shape),'arms':alternatives})
    return {
        'seed':seed,'repetitions':repetitions,
        'python':sys.version.split()[0],'platform':platform.platform(),
        'numpy':np.__version__,'scipy':scipy.__version__,
        'max_pairs':100,'max_support':2,'max_operations':1000000,
        'interpretation':'time includes preprocessing; operation count includes mixed-pruning LP pivots; no SOTA claim',
        'results':outcomes,
    }


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
