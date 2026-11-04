"""v55 matched workload: exact direct-witness probe vs legacy rational simplex.

Run from backend/: python benchmark_exact_probe_v55.py --repetitions 3
Both arms traverse identical support prefixes with the same budgets,
payoff matrices, pruning mode, and exact LP implementation. Evidence
documents distributional runtime and regressions; no SOTA claim.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from fractions import Fraction
import numpy as np
from general_exact_search import search_exact_nash_cells
from exact_support import _verify_exact_equilibrium


def fixtures(seed=55055):
    rng = np.random.default_rng(seed)
    n = 4
    z = np.zeros((n,n))
    descending = np.arange(n,0,-1)
    dominance_a = np.tile(descending[:,None], (1,n))
    dominance_b = np.tile(descending[None,:], (n,1))
    identity = 3*np.eye(n)
    cyclic = np.fromfunction(lambda i,j: (i*3+j*5)%7, (n,n))
    return [
        ('all_ties_4x4', z, z, 225, 4),
        ('dominance_4x4', dominance_a, dominance_b, 225, 4),
        ('coordination_4x4', identity, identity, 225, 4),
        ('cyclic_4x4', cyclic, cyclic.T, 225, 4),
        ('random_4x4', rng.integers(-4,5,(4,4)), rng.integers(-4,5,(4,4)), 225, 4),
        ('random_6x6_prefix', rng.integers(-4,5,(6,6)), rng.integers(-4,5,(6,6)), 120, 2),
        ('tie_heavy_6x6_prefix', np.zeros((6,6)), np.zeros((6,6)), 120, 2),
    ]


def _signature(out):
    return {(tuple(item['support']['row']),tuple(item['support']['col']))
            for item in out['families']}


def _verify(a,b,result):
    aa = [[Fraction(str(float(v))) for v in row] for row in a]
    bb = [[Fraction(str(float(v))) for v in row] for row in b]
    for item in result['families']:
        x = [Fraction(z) for z in item['row_probabilities_exact']]
        y = [Fraction(z) for z in item['col_probabilities_exact']]
        if not _verify_exact_equilibrium(aa, bb, x, y):
            raise AssertionError('Benchmark produced invalid exact Nash profile')


def run(repetitions=3, seed=55055):
    if type(repetitions) is not int or not 1 <= repetitions <= 20:
        raise ValueError('repetitions must be integer 1..20')
    data = fixtures(seed)
    outputs = []
    for name, a, b, pairs, max_support in data:
        settings = dict(max_pairs=pairs, max_support=max_support,
                        max_operations=2000000, prune_supports=False,
                        mixed_pruning=False, feasibility_backend='simplex')
        timing = {False: [], True: []}
        baseline = quick = None
        for rep in range(repetitions):
            # Alternate paired evaluation order to reduce order-dependent drift.
            for enabled in ((False, True) if rep % 2 == 0 else (True, False)):
                start = time.perf_counter()
                result = search_exact_nash_cells(
                    a, b, feasibility_fast_path=enabled, **settings)
                timing[enabled].append(time.perf_counter() - start)
                if enabled:
                    quick = result
                else:
                    baseline = result
        if baseline is None or quick is None:
            raise AssertionError('No benchmark outputs')
        if (_signature(baseline) != _signature(quick) or
                baseline['support_pairs_checked'] != quick['support_pairs_checked'] or
                baseline['incomplete_reason'] != quick['incomplete_reason'] or
                baseline['complete_equilibrium_set'] != quick['complete_equilibrium_set']):
            raise AssertionError(f'Feasibility optimization changed Nash-set classification: {name}')
        _verify(a,b,baseline)
        _verify(a,b,quick)
        outputs.append({
            'case': name, 'shape': list(a.shape),
            'max_pairs': pairs, 'max_support': max_support,
            'support_pairs_checked': quick['support_pairs_checked'],
            'complete_equilibrium_set': quick['complete_equilibrium_set'],
            'nash_families': len(quick['families']),
            'baseline_lp_calls': baseline['simplex_lp_calls'],
            'optimized_lp_calls': quick['simplex_lp_calls'],
            'optimized_uniform_witness_hits': quick['uniform_witness_hits'],
            'optimized_singleton_infeasible_hits': quick['singleton_infeasible_hits'],
            'optimized_probe_operations': quick['feasibility_probe_operations'],
            'baseline_total_work': baseline['exact_operations'],
            'optimized_total_work': quick['exact_operations'],
            'baseline_all_seconds': timing[False],
            'optimized_all_seconds': timing[True],
            'baseline_median_seconds': statistics.median(timing[False]),
            'optimized_median_seconds': statistics.median(timing[True]),
            'observed_median_ratio_baseline_over_optimized':
                statistics.median(timing[False])/statistics.median(timing[True]),
        })
    return {
        'campaign': 'v55 exact feasibility uniform-witness and singleton shortcut',
        'seed': seed, 'repetitions': repetitions,
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'numpy_version': np.__version__,
        'workload': 'same ordered support prefixes, no conditional pruning; matched matrix, LP backend, and budgets',
        'work_units': 'pivot tableau cell updates plus exact-payoff coefficient probe inspections; not a measure of fraction bit complexity',
        'limitations': 'Small deterministic sample; CI contention and CPU variation prevent general speed claims; may regress on random cases; raw work units count different operations.',
        'cases': outputs,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--seed',type=int,default=55055)
    p.add_argument('--output',default='benchmark_exact_probe_v55_results.json')
    opts=p.parse_args(argv)
    out=run(opts.repetitions,opts.seed)
    with open(opts.output,'w',encoding='utf-8') as f:
        json.dump(out,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({'cases':len(out['cases']),
                      'all_verified':True,
                      'ratios':{c['case']:round(c['observed_median_ratio_baseline_over_optimized'],3)
                                for c in out['cases']},
                      'artifact':opts.output},indent=2))


if __name__=='__main__':
    main()
