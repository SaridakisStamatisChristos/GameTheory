"""v55 reproducible paired exact-feasibility scaling experiment."""
import argparse
import json
import platform
import statistics
import time
from hashlib import sha256
import numpy as np
from general_exact_search import search_exact_nash_cells


def fixtures(seed=55055):
    rng=np.random.default_rng(seed)
    cases=[]
    for n in (2,3,4,6,8):
        zero=np.zeros((n,n),dtype=int)
        cases.append((f'all_ties_{n}',zero,zero.copy()))
        a=np.tile(np.arange(n,0,-1,dtype=int)[:,None],(1,n))
        b=np.tile(np.arange(n,0,-1,dtype=int)[None,:],(n,1))
        cases.append((f'dominated_{n}',a,b))
        cases.append((f'random_{n}',rng.integers(-7,8,(n,n)),rng.integers(-7,8,(n,n))))
    return cases


def signature(out):
    return sorted((tuple(f['support']['row']),tuple(f['support']['col']),
                   tuple(f['row_probabilities_exact']),
                   tuple(f['col_probabilities_exact']))
                  for f in out['families'])


def run(repetitions=3,seed=55055):
    if type(repetitions) is not int or not 1<=repetitions<=10:
        raise ValueError('repetitions must be 1..10')
    games=fixtures(seed)
    digest=sha256(json.dumps([(name,a.tolist(),b.tolist()) for name,a,b in games],
                             separators=(',',':')).encode()).hexdigest()
    results=[]
    for name,a,b in games:
        m,n=a.shape
        total=(2**m-1)*(2**n-1)
        max_pairs=min(total,96)
        settings=dict(max_pairs=max_pairs,max_support=min(m,n,3),
                      max_operations=2000000,feasibility_backend='simplex',
                      prune_supports=False,mixed_pruning=False)
        measurements={}
        for fast in (False,True):
            elapsed=[];outputs=[]
            for _ in range(repetitions):
                start=time.perf_counter()
                out=search_exact_nash_cells(a,b,feasibility_fast_path=fast,**settings)
                elapsed.append(time.perf_counter()-start)
                outputs.append(out)
            if any(signature(o)!=signature(outputs[0]) for o in outputs):
                raise AssertionError('Non-deterministic support witnesses')
            measurements[fast]={'time':elapsed,'out':outputs[0]}
        old,new=measurements[False]['out'],measurements[True]['out']
        # Feasible cells are invariant; witnesses may differ inside continuous cells.
        supports=lambda out:sorted((tuple(f['support']['row']),tuple(f['support']['col']))
                                   for f in out['families'])
        for field in ('support_pairs_checked','complete_equilibrium_set',
                      'incomplete_reason','feasible_cells_found'):
            if old[field]!=new[field]:
                raise AssertionError(f'Feasibility path altered {field}: {name}')
        if supports(old)!=supports(new):
            raise AssertionError(f'Feasibility path changed Nash support set: {name}')
        results.append({'case':name,'shape':[m,n],'support_pairs_total':total,
                        'support_pairs_checked':new['support_pairs_checked'],
                        'complete':new['complete_equilibrium_set'],
                        'baseline_median_seconds':statistics.median(measurements[False]['time']),
                        'fast_median_seconds':statistics.median(measurements[True]['time']),
                        'baseline_seconds':measurements[False]['time'],
                        'fast_seconds':measurements[True]['time'],
                        'baseline_operations':old['exact_operations'],
                        'fast_operations':new['exact_operations'],
                        'baseline_lp_calls':old['simplex_lp_calls'],
                        'fast_lp_calls':new['simplex_lp_calls'],
                        'uniform_hits':new['uniform_witness_hits'],
                        'singleton_rejections':new['singleton_infeasible_hits'],
                        'support_sets_identical':True})
    return {'version':'v55','seed':seed,'repetitions':repetitions,
            'fixture_sha256':digest,'python':platform.python_version(),
            'numpy':np.__version__,'cases':results,
            'limitations':'Paired measurements are environment-dependent; rational bit complexity and peak RSS are not measured. Partial searches are labeled and no global SOTA claim is made.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--seed',type=int,default=55055)
    parser.add_argument('--output',default='benchmark_exact_scaling_v55_results.json')
    args=parser.parse_args()
    result=run(args.repetitions,args.seed)
    with open(args.output,'w',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'cases_checked':len(result['cases']),
                      'all_support_sets_identical':all(c['support_sets_identical'] for c in result['cases']),
                      'output':args.output}))


if __name__=='__main__':
    main()
