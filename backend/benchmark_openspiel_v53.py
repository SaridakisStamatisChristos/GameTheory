"""Reproducible OpenSpiel 2.0.2 numerical LP vs exact-rational minimax comparison.

The reference uses OpenSpiel's own matrix-game model and LP solver, not a
hand-written SciPy surrogate. Strict failure when unavailable or invalid.
"""
import argparse
from fractions import Fraction
import json
import platform
import statistics
import sys
import time
from importlib.metadata import version
import numpy as np
from benchmark_gambit_v52 import fixtures as gambit_fixtures, fixture_digest
from exact_minimax import solve_exact_zero_sum


def fixtures(seed=53053):
    """Reuse the complete v52 fixture universe, add rectangular and 10x10 games."""
    games = gambit_fixtures(seed=52052)
    rng = np.random.default_rng(seed)
    games += [
        ('rectangular_4x7', rng.integers(-10, 11, size=(4, 7), dtype=np.int64)),
        ('rectangular_7x4', rng.integers(-10, 11, size=(7, 4), dtype=np.int64)),
        ('random_10x10', rng.integers(-8, 9, size=(10, 10), dtype=np.int64)),
    ]
    return games


def model_translation(a):
    """Exact integer translation preserves all equilibrium strategies.

    Keeping the LP coefficients close to zero avoids treating a large
    common payoff offset as a strategic conditioning problem. Timings
    are not strictly identical-input speed comparisons because only
    OpenSpiel's LP receives translated payoffs.
    """
    a = np.asarray(a)
    if (a.ndim != 2 or a.size == 0 or not np.issubdtype(a.dtype, np.integer)
            or a.size > 100 or np.max(np.abs(a)) > 1000000):
        raise ValueError('Requires nonempty bounded integer zero-sum payoff matrix')
    offset = int(a.min())
    model = a.astype(np.int64) - offset
    if np.max(np.abs(model)) > 1000000:
        raise ValueError('Translated payoff exceeds numerical fixture budget')
    return model, offset


def independent_numeric_saddle(a, x, y, reference_value, row_value, col_value, *,
                               tolerance=2e-5):
    """Audit the original integer game without either solver's internal code."""
    a = np.asarray(a, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (a.ndim != 2 or a.size == 0 or x.shape != (a.shape[0],) or
            y.shape != (a.shape[1],) or
            not all(np.isfinite(v).all() for v in (a, x, y)) or
            not np.isfinite([reference_value, row_value, col_value, tolerance]).all()
            or tolerance <= 0):
        raise AssertionError('Invalid numerical equilibrium certificate inputs')
    simplex_error = max(abs(float(x.sum()) - 1), abs(float(y.sum()) - 1),
                        max(0.0, -float(x.min()), -float(y.min())))
    lower = float(np.min(x @ a))
    upper = float(np.max(a @ y))
    expected = float(x @ a @ y)
    gap = max(0.0, upper - lower)
    values = [lower, upper, expected, float(row_value), -float(col_value)]
    value_error = max(abs(v - float(reference_value)) for v in values)
    if (simplex_error > 1e-7 or gap > tolerance or value_error > tolerance):
        raise AssertionError(
            f'OpenSpiel failed original-payoff Nash audit: '
            f'simplex={simplex_error}, gap={gap}, value_error={value_error}')
    return {'simplex_error': simplex_error,
            'unilateral_gap_original_units': gap,
            'max_value_error_original_units': value_error,
            'reference_value_original_units': float(reference_value)}


def compare_case(pyspiel, lp_solver, name, a, repetitions):
    if type(repetitions) is not int or not 1 <= repetitions <= 30:
        raise ValueError('repetitions must be a Python integer in 1..30')
    original = np.asarray(a, dtype=np.int64)
    transformed, offset = model_translation(original)
    # Verify every coefficient of the game submitted to OpenSpiel.
    model = transformed.astype(float).tolist()
    neg = (-transformed).astype(float).tolist()
    exact_times, reference_times = [], []
    last_cert = None
    value = None
    for _ in range(repetitions):
        start = time.perf_counter()
        ours = solve_exact_zero_sum(original)
        exact_value = Fraction(ours['value_exact'])
        exact_times.append(time.perf_counter()-start)
        start = time.perf_counter()
        game = pyspiel.create_matrix_game(model, neg)
        if (game.num_rows() != original.shape[0] or
                game.num_cols() != original.shape[1] or
                any(game.player_utility(0, i, j) != float(transformed[i, j]) or
                    game.player_utility(1, i, j) != float(-transformed[i, j])
                    for i in range(original.shape[0]) for j in range(original.shape[1]))):
            raise AssertionError('OpenSpiel matrix game changed model payoffs')
        x, y, p0, p1 = lp_solver.solve_zero_sum_matrix_game(game)
        reference_times.append(time.perf_counter()-start)
        last_cert = independent_numeric_saddle(
            original, x, y, exact_value, float(p0)+offset, float(p1)-offset)
        if value is not None and value != exact_value:
            raise AssertionError('Exact solver was inconsistent between repeats')
        value = exact_value
    return {
        'name': name, 'shape': list(original.shape),
        'exact_value': str(value), 'open_spiel_payoff_translation': offset,
        'open_spiel_certificate': last_cert,
        'exact_times_seconds': exact_times,
        'open_spiel_times_seconds': reference_times,
        'median_exact_seconds': statistics.median(exact_times),
        'median_open_spiel_seconds': statistics.median(reference_times),
        'samples_per_engine': repetitions,
    }


def run(repetitions=3, seed=53053):
    if type(repetitions) is not int or not 1 <= repetitions <= 30:
        raise ValueError('repetitions must be a Python integer in 1..30')
    import pyspiel
    import cvxpy as cp
    from open_spiel.python.algorithms import lp_solver
    # No fake reference: use OpenSpiel's original LP implementation and ECOS.
    if 'ECOS' not in cp.installed_solvers():
        raise RuntimeError('OpenSpiel LP requires an available ECOS numerical solver')
    games = fixtures(seed)
    cases = [compare_case(pyspiel, lp_solver, name, a, repetitions) for name, a in games]
    return {
        'campaign': 'v53 independent OpenSpiel zero-sum numerical LP',
        'source': 'open_spiel.python.algorithms.lp_solver.solve_zero_sum_matrix_game',
        'result': 'all_numerical_candidates_audited',
        'model': 'integer zero-sum, OpenSpiel uses integer payoff translation',
        'seed': seed, 'fixture_digest_sha256': fixture_digest(games),
        'samples_per_engine': repetitions, 'cases': cases,
        'open_spiel_version': version('open-spiel'),
        'cvxpy_version': version('cvxpy'),
        'ecos_version': version('ecos'),
        'numpy_version': np.__version__,
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'assertion_tolerance_original_units': 2e-5,
        'scope_note': 'Numerical vs exact methods; timings include different preprocessing and unequal model translations. No speedup or SOTA inference.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--seed', type=int, default=53053)
    parser.add_argument('--output', default='benchmark_openspiel_v53_results.json')
    args = parser.parse_args(argv)
    results = run(repetitions=args.repetitions, seed=args.seed)
    # An artifact exists only after ALL external comparisons passed.
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({
        'cases_checked': len(results['cases']),
        'samples_per_engine': results['samples_per_engine'],
        'all_certified': True, 'open_spiel_version': results['open_spiel_version'],
        'fixture_digest_sha256': results['fixture_digest_sha256'],
        'artifact_path': args.output,
    }, indent=2))


if __name__ == '__main__':
    main()
