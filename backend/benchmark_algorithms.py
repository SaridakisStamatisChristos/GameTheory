"""Reproducible algorithm correctness/performance campaign. Run: python benchmark_algorithms.py"""
import argparse
import json
import time
import numpy as np
from scipy.optimize import linprog
from advanced import support_equilibria
from approximate import approximate_nash
from main import zero_sum, ZeroSum


def nash_gap(a, b, x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    assert x.shape == (a.shape[0],) and y.shape == (a.shape[1],)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    assert np.min(x) >= -1e-7 and np.min(y) >= -1e-7
    assert abs(x.sum()-1) < 1e-6 and abs(y.sum()-1) < 1e-6
    return float(max(0., np.max(a@y)-x@a@y, np.max(x@b)-x@b@y))


def run(seed=1729, repeats=5, iterations=1000):
    rng = np.random.default_rng(seed)
    cases = [
        ('prisoners_dilemma', [[3, 0], [5, 1]], [[3, 5], [0, 1]]),
        ('matching_pennies', [[1, -1], [-1, 1]], [[-1, 1], [1, -1]]),
        ('rock_paper_scissors', [[0, -1, 1], [1, 0, -1], [-1, 1, 0]], [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]),
        ('degenerate_constant', [[0, 0], [0, 0]], [[0, 0], [0, 0]]),
    ]
    for n in [2, 3, 5, 10]:
        a = rng.normal(size=(n,n)); cases.append((f'random_zero_sum_{n}', a.tolist(), (-a).tolist()))
    records = []
    for name, aa, bb in cases:
        a, b = np.asarray(aa, float), np.asarray(bb, float)
        for algorithm in ['support', 'approximate'] + (['zero_sum_lp'] if np.allclose(a, -b) else []):
            durations = []; gaps = []; count = 0; failures = []
            for repeat in range(repeats):
                start = time.perf_counter_ns()
                try:
                    if algorithm == 'support':
                        solutions = support_equilibria(a,b)
                        profiles = [(s['row_probabilities'], s['col_probabilities']) for s in solutions]
                    elif algorithm == 'approximate':
                        r = approximate_nash(a,b,iterations,2,seed+repeat)
                        # API contract is checked explicitly; no silent field guessing.
                        profiles = [(r['profile']['row_probabilities'],r['profile']['col_probabilities'])]
                    else:
                        r = zero_sum(ZeroSum(payoffs=aa))
                        profiles = [(r['row_strategy'],r['col_strategy'])]
                    durations.append((time.perf_counter_ns()-start)/1e6)
                    count += len(profiles)
                    gaps.extend(nash_gap(a,b,x,y) for x,y in profiles)
                except Exception as exc:
                    failures.append(f'{type(exc).__name__}: {exc}')
            records.append({'case':name,'algorithm':algorithm,'repeats':repeats,'solutions':count,
                            'median_ms':float(np.median(durations)) if durations else None,
                            'p95_ms':float(np.percentile(durations,95)) if durations else None,
                            'worst_nash_gap':max(gaps) if gaps else None,
                            'failures':failures})
    return {'seed':seed,'iterations':iterations,'records':records,
            'note':'Timings include Python solver overhead and are environment-specific. Support enumeration is incomplete for degenerate games.'}

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--output',default='benchmark-results.json');args=p.parse_args()
    result=run(args.seed,args.repeats,args.iterations)
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
