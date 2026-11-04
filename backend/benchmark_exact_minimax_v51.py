"""Paired zero-sum minimax smoke comparison; no cross-engine superiority claim."""
import json
import platform
import statistics
import sys
from time import perf_counter
from fractions import Fraction
import numpy as np
import scipy
from scipy.optimize import linprog
from exact_minimax import solve_exact_zero_sum


def run(repetitions=3, seed=51051):
    if type(repetitions) is not int or not 1 <= repetitions <= 20:
        raise ValueError('repetitions must be 1..20')
    rng = np.random.default_rng(seed)
    games = [
        ('matching_pennies', np.asarray([[1, -1], [-1, 1]], float)),
        ('rock_paper_scissors', np.asarray([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)),
        ('zero_ties_4x4', np.zeros((4, 4))),
    ]
    for n in (3, 5, 8):
        for i in range(3):
            games.append((f'random_{n}_{i}', rng.integers(-9, 10, size=(n, n)).astype(float)))
    results = []
    for name, a in games:
        m, n = a.shape
        exact_times, scipy_times = [], []
        witness = None
        for _ in range(repetitions):
            start = perf_counter()
            witness = solve_exact_zero_sum(a)
            exact_times.append(perf_counter()-start)
            start = perf_counter()
            ref = linprog(np.r_[np.zeros(m), -1.],
                          A_ub=np.c_[-a.T, np.ones(n)], b_ub=np.zeros(n),
                          A_eq=np.r_[np.ones(m), 0.][None, :], b_eq=[1],
                          bounds=[(0, None)]*m+[(None, None)], method='highs')
            scipy_times.append(perf_counter()-start)
            if not ref.success or abs(float(Fraction(witness['value_exact']))-ref.x[-1]) > 1e-8:
                raise AssertionError('Independent minimax reference disagrees')
        results.append({
            'case': name, 'shape': list(a.shape),
            'exact_value': witness['value_exact'],
            'exact_operations': witness['exact_operations'],
            'median_exact_seconds': statistics.median(exact_times),
            'median_scipy_seconds': statistics.median(scipy_times),
            'repetitions': repetitions, 'reference_valid': True,
        })
    return {
        'comparison': 'rational primal-dual simplex vs numerical SciPy/HiGHS (different precision semantics)',
        'seed': seed, 'repetitions': repetitions,
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'gambit': 'not_run', 'open_spiel': 'not_run',
        'limitations': 'Smoke-scale timings are descriptive only; no cross-engine speedup or SOTA conclusion.',
        'cases': results,
    }


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
