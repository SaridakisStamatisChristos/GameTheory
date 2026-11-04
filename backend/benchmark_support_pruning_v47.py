"""Reproducible v47 exact-support pruning differential benchmark.

Usage from backend/: python benchmark_support_pruning_v47.py
Compares identical ordered support prefixes and exact LP backend.
Timings are machine-specific; budget and support order are held constant.
"""
import json
import statistics
import time
import numpy as np
from general_exact_search import search_exact_nash_cells


def _signature(result):
    return sorted((tuple(x['support']['row']), tuple(x['support']['col']))
                  for x in result['families'])


def run(repetitions=3, seed=4701):
    rng=np.random.default_rng(seed)
    cases=[]
    for size in (4,6,8,10):
        # Globally dominated, cyclic, and unstructured games exercise
        # very different pruning rates. Don't cherry-pick only wins.
        descending=np.arange(size,0,-1,dtype=float)
        a=np.tile(descending[:,None],(1,size))
        b=np.tile(descending[None,:],(size,1))
        cases.append((f'globally_dominated_{size}',a,b))
        a=np.fromfunction(lambda i,j:(i+j)%size,(size,size),dtype=int)
        b=np.fromfunction(lambda i,j:(i*3+j*5)%size,(size,size),dtype=int)
        cases.append((f'cyclic_{size}',a,b))
        a=rng.integers(-4,5,(size,size))
        b=rng.integers(-4,5,(size,size))
        cases.append((f'seeded_random_{size}',a,b))
    outcomes=[]
    for name,a,b in cases:
        alternatives={}
        for pruning in (False,True):
            times=[];last=None
            for _ in range(repetitions):
                before=time.perf_counter()
                result=search_exact_nash_cells(
                    a,b,max_pairs=128,max_support=2,max_operations=1000000,
                    feasibility_backend='simplex',prune_supports=pruning)
                times.append(time.perf_counter()-before)
                if last is not None and _signature(result)!=_signature(last):
                    raise AssertionError('Non-deterministic exact Nash supports')
                last=result
            alternatives[pruning]={'median_seconds':statistics.median(times),
                                  'result':last}
        base=alternatives[False]['result']
        opt=alternatives[True]['result']
        if (_signature(base)!=_signature(opt) or
            base['support_pairs_checked']!=opt['support_pairs_checked'] or
            base['complete_equilibrium_set']!=opt['complete_equilibrium_set'] or
            base['incomplete_reason']!=opt['incomplete_reason']):
            raise AssertionError('Pruning altered mathematical support classification')
        outcomes.append({
            'case':name,
            'shape':list(a.shape),
            'baseline_median_seconds':alternatives[False]['median_seconds'],
            'pruned_median_seconds':alternatives[True]['median_seconds'],
            'support_pairs_checked':opt['support_pairs_checked'],
            'feasible_support_pairs':opt['feasible_cells_found'],
            'dominance_pruned_pairs':opt['dominance_pruned_pairs'],
            'baseline_exact_operations':base['exact_operations'],
            'pruned_exact_operations':opt['exact_operations'],
            'completeness_status':opt['status'],
        })
    return {'seed':seed,'repetitions':repetitions,
            'pair_budget':128,'max_support':2,
            'exact_backend':'simplex','results':outcomes,
            'caution':'No general speedup claim; preprocessing costs and workload sensitivity are included in timings.'}


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
