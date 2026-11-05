"""v56 paired exact sign-screen differential, CPU and resource campaign.

No timing claims without equivalent support sets, complete flags, and identical
input order. Tracemalloc peak is Python-traced allocation, not process RSS.
"""
import argparse
from hashlib import sha256
import json
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np
from general_exact_search import search_exact_nash_cells


def fixtures(seed=56056):
    rng = np.random.default_rng(seed)
    games = []
    for size in (2, 3, 4, 6, 8):
        z = np.zeros((size, size), dtype=int)
        ranked = np.arange(size, 0, -1, dtype=int)
        games.append((f'all_ties_{size}', z, z.copy()))
        games.append((f'weak_dominated_{size}',
                      np.tile(ranked[:, None], (1,size)),
                      np.tile(ranked[None,:], (size,1))))
        games.append((f'random_{size}',
                      rng.integers(-7,8,(size,size)),
                      rng.integers(-7,8,(size,size))))
    return games


def _signature(out):
    return sorted((tuple(f['support']['row']),tuple(f['support']['col']))
                  for f in out['families'])


def run(repetitions=3,seed=56056):
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError('repetitions must be integer 1..10')
    games = fixtures(seed)
    fixture_bytes = json.dumps([(name,a.tolist(),b.tolist())
                                for name,a,b in games],separators=(',',':')).encode()
    cases = []
    for name,a,b in games:
        m,n = a.shape
        total = (2**m-1)*(2**n-1)
        settings = dict(max_pairs=min(total,96),max_support=min(m,n,3),
                        max_operations=2000000,feasibility_backend='simplex',
                        feasibility_fast_path=True,prune_supports=False,
                        mixed_pruning=False)
        measurements = {}
        for enabled in (False,True):
            durations = []
            last = None
            for _ in range(repetitions):
                start = time.perf_counter()
                output = search_exact_nash_cells(
                    a,b,feasibility_sign_screen=enabled,**settings)
                durations.append(time.perf_counter()-start)
                if last is not None and _signature(last) != _signature(output):
                    raise AssertionError(f'Non-deterministic v56 support set: {name}')
                last = output
            tracemalloc.start()
            try:
                sampled = search_exact_nash_cells(
                    a,b,feasibility_sign_screen=enabled,
                    track_rational_bits=True,**settings)
                _,peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            if _signature(sampled) != _signature(last):
                raise AssertionError(f'v56 opt-in bit diagnostics changed feasible supports: {name}')
            measurements[enabled] = {'out':last,'seconds':durations,
                                      'traced_peak_bytes':peak,
                                      'sample_num_bits':sampled['max_tableau_numerator_bits'],
                                      'sample_den_bits':sampled['max_tableau_denominator_bits']}
        old,new = measurements[False]['out'],measurements[True]['out']
        for key in ('complete_equilibrium_set','incomplete_reason',
                    'support_pairs_checked','feasible_cells_found'):
            if old[key] != new[key]:
                raise AssertionError(f'v56 screen changed search contract for {name}: {key}')
        if _signature(old) != _signature(new):
            raise AssertionError(f'v56 screen changed exact support set: {name}')
        if new['simplex_lp_calls'] > old['simplex_lp_calls']:
            raise AssertionError(f'v56 screen introduced additional LP calls: {name}')
        cases.append({
            'case':name,'shape':[m,n],
            'support_pairs_total':total,
            'support_pairs_checked':new['support_pairs_checked'],
            'complete':new['complete_equilibrium_set'],
            'incomplete_reason':new['incomplete_reason'],
            'exact_support_sets_equal':True,
            'screen_rejections':new['sign_screen_rejections'],
            'screen_operations':new['sign_screen_operations'],
            'baseline_simplex_lp_calls':old['simplex_lp_calls'],
            'screen_simplex_lp_calls':new['simplex_lp_calls'],
            'baseline_exact_operations':old['exact_operations'],
            'screen_exact_operations':new['exact_operations'],
            'baseline_seconds':measurements[False]['seconds'],
            'screen_seconds':measurements[True]['seconds'],
            'baseline_median_seconds':statistics.median(measurements[False]['seconds']),
            'screen_median_seconds':statistics.median(measurements[True]['seconds']),
            'baseline_tracemalloc_peak_bytes':measurements[False]['traced_peak_bytes'],
            'screen_tracemalloc_peak_bytes':measurements[True]['traced_peak_bytes'],
            'baseline_tableau_numerator_bits':measurements[False]['sample_num_bits'],
            'baseline_tableau_denominator_bits':measurements[False]['sample_den_bits'],
            'screen_tableau_numerator_bits':measurements[True]['sample_num_bits'],
            'screen_tableau_denominator_bits':measurements[True]['sample_den_bits'],
        })
    return {'release':'v56','seed':seed,'repetitions':repetitions,
            'fixture_sha256':sha256(fixture_bytes).hexdigest(),
            'python':sys.version.split()[0], 'platform':platform.platform(),
            'numpy':np.__version__, 'cases':cases,
            'resource_method':'tracemalloc peak tracked Python allocations in separate untimed opt-in bit-tracking runs; not RSS',
            'timing_method':'independent untraced runs in identical case order; sample order baseline then screen; host-specific',
            'limitations':'Charged arithmetic operations are not actual bit complexity. Benchmarks use bounded (often incomplete) support prefixes and do not establish general speedup, completeness for larger games, or global SOTA.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--seed',type=int,default=56056)
    parser.add_argument('--output',default='benchmark_sign_screen_v56_results.json')
    args=parser.parse_args()
    result = run(repetitions=args.repetitions,seed=args.seed)
    with open(args.output,'w',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps({'cases_checked':len(result['cases']),
                      'all_exact_support_sets_equal':all(
                          c['exact_support_sets_equal'] for c in result['cases']),
                      'total_sign_screen_rejections':sum(
                          c['screen_rejections'] for c in result['cases']),
                      'total_lp_calls_saved':sum(
                          c['baseline_simplex_lp_calls']-c['screen_simplex_lp_calls']
                          for c in result['cases']),
                      'output':args.output},indent=2))


if __name__ == '__main__':
    main()
