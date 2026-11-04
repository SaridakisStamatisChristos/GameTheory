"""Independent exact-rational Gambit LP comparison for integer zero-sum matrix games.

Requires pinned pygambit==17.0.0b2, installed ONLY in its dedicated CI job.
This benchmark is a reproducible correctness check, not a SOTA speed claim.
"""
import argparse
from fractions import Fraction
import hashlib
from io import BytesIO
from importlib.metadata import version
import json
import platform
import statistics
import sys
from time import perf_counter
import numpy as np
from exact_minimax import solve_exact_zero_sum


def fixtures(seed=52052):
    rng = np.random.default_rng(seed)
    matrices = [
        ('matching_pennies', [[1, -1], [-1, 1]]),
        ('rock_paper_scissors', [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
        ('all_zero_3x3', [[0]*3 for _ in range(3)]),
        ('constant_2x4', [[4]*4 for _ in range(2)]),
        ('rectangular_2x3', [[-2, 1, 3], [2, -4, 0]]),
        ('rectangular_3x2', [[2, -5], [-1, 3], [4, 0]]),
        ('pure_saddle_3x3', [[3, 2, 1], [0, -1, -2], [1, 0, -1]]),
        ('near_tie_integer_2x2', [[100000, 99999], [99999, 100000]]),
    ]
    for n, count in ((3, 3), (5, 3), (8, 2)):
        for k in range(count):
            matrices.append((f'random_{n}x{n}_{k}', rng.integers(
                -10, 11, size=(n, n)).tolist()))
    return [(name, np.asarray(a, dtype=np.int64)) for name, a in matrices]


def exact_certificate(a, x, y):
    """Direct saddle-point audit; completely independent of both solver internals."""
    a = np.asarray(a)
    if a.ndim != 2 or not a.size or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('Comparator needs a finite integer payoff matrix')
    m, n = a.shape
    if len(x) != m or len(y) != n:
        raise AssertionError('Reference strategy dimensions disagree with the payoff matrix')
    xx = [Fraction(p) for p in x]
    yy = [Fraction(p) for p in y]
    if (any(p < 0 for p in (*xx, *yy)) or
            sum(xx) != 1 or sum(yy) != 1):
        raise AssertionError('Reference strategy does not belong to the simplex')
    lower = min(sum(xx[i]*int(a[i, j]) for i in range(m)) for j in range(n))
    upper = max(sum(int(a[i, j])*yy[j] for j in range(n)) for i in range(m))
    if lower != upper:
        raise AssertionError('Reference failed exact zero-sum saddle-point inequalities')
    expectation = sum(xx[i]*int(a[i, j])*yy[j] for i in range(m) for j in range(n))
    if expectation != lower:
        raise AssertionError('Reference expected payoff disagrees with the game value')
    return lower


def fixture_digest(games):
    payload = [{'name': name, 'matrix': a.tolist()} for name, a in games]
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def nfg_bytes(a):
    """Standard Gambit NFG payoff format: first player's action varies fastest."""
    a = np.asarray(a)
    if a.ndim != 2 or a.size == 0 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('NFG comparator requires a nonempty integer matrix')
    m, n = a.shape
    entries = ' '.join(f'{int(a[i, j])} {-int(a[i, j])}'
                       for j in range(n) for i in range(m))
    return (f'NFG 1 R "GameTheory v52" {{ "Row" "Column" }} '
            f'{{ {m} {n} }}\\n\\n{entries}\\n').replace('\\n', '\n').encode('ascii')


def compare_case(gbt, name, a, repetitions):
    if not 1 <= repetitions <= 20:
        raise ValueError('repetitions must be 1..20')
    if a.size > 100 or np.max(np.abs(a)) > 1000000:
        raise ValueError('Benchmark fixtures must be exactly representable integer matrices')
    ours_seconds, gambit_seconds = [], []
    ours_value = gambit_value = None
    ours_strategy = gambit_strategy = None
    for _ in range(repetitions):
        # Both timings include their public-facing solve and output extraction.
        start = perf_counter()
        ours = solve_exact_zero_sum(a)
        x_ours = [Fraction(z) for z in ours['row_strategy_exact']]
        y_ours = [Fraction(z) for z in ours['col_strategy_exact']]
        v_ours = exact_certificate(a, x_ours, y_ours)
        ours_seconds.append(perf_counter() - start)
        if v_ours != Fraction(ours['value_exact']):
            raise AssertionError('GameTheory value disagrees with independent certificate')

        start = perf_counter()
        game = gbt.read_nfg(BytesIO(nfg_bytes(a)))
        game_payoffs = game.to_arrays()
        if (len(game_payoffs) != 2 or
                any(Fraction(str(game_payoffs[0][i, j])) != int(a[i, j])
                    or Fraction(str(game_payoffs[1][i, j])) != -int(a[i, j])
                    for i in range(a.shape[0]) for j in range(a.shape[1]))):
            raise AssertionError('Gambit input round trip changed exact payoff values')
        answer = gbt.nash.lp_solve(game, rational=True, use_strategic=True)
        if not answer.equilibria:
            raise AssertionError('Gambit exact LP returned no equilibrium')
        profile = answer.equilibria[0]
        players = list(game.players)
        if len(players) != 2:
            raise AssertionError('Gambit did not create a two-player game')
        x_gambit = [Fraction(str(profile[players[0]][strategy]))
                    for strategy in game.get_strategies(players[0])]
        y_gambit = [Fraction(str(profile[players[1]][strategy]))
                    for strategy in game.get_strategies(players[1])]
        v_gambit = exact_certificate(a, x_gambit, y_gambit)
        gambit_seconds.append(perf_counter() - start)
        if v_gambit != v_ours:
            raise AssertionError('Gambit and GameTheory exact game values disagree')
        if ours_value is not None and ours_value != v_ours:
            raise AssertionError('GameTheory produced non-reproducible exact values')
        if gambit_value is not None and gambit_value != v_gambit:
            raise AssertionError('Gambit produced non-reproducible exact values')
        ours_value, gambit_value = v_ours, v_gambit
        ours_strategy = (x_ours, y_ours)
        gambit_strategy = (x_gambit, y_gambit)
    return {
        'case': name, 'shape': list(a.shape),
        'game_value_exact': str(ours_value),
        'strategy_profiles_equal': ours_strategy == gambit_strategy,
        'both_exact_certificates_passed': True,
        'game_theory_median_seconds': statistics.median(ours_seconds),
        'gambit_median_seconds': statistics.median(gambit_seconds),
        'game_theory_all_seconds': ours_seconds,
        'gambit_all_seconds': gambit_seconds,
    }


def run(repetitions=3, seed=52052):
    if type(repetitions) is not int or not 1 <= repetitions <= 20:
        raise ValueError('repetitions must be 1..20')
    import pygambit as gbt  # External comparator is required here, never skipped.
    games = fixtures(seed)
    return {
        'research_gate': 'v52 independent Gambit exact-rational LP',
        'evidence': 'executed_external_engine',
        'seed': seed, 'repetitions': repetitions,
        'fixtures_sha256': fixture_digest(games),
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pygambit_version': version('pygambit'),
        'gambit_solver': 'pygambit.nash.lp_solve(rational=True,use_strategic=True)',
        'inputs': 'integer payoffs exactly represented in both systems; all values <= 1e6',
        'comparison_metric': 'exact game value and independent exact saddle-point certificate',
        'timing_scope': 'public solve, NFG import and returned strategy extraction/audit',
        'limitations': ('Different implementations and preprocessing overheads; '
                        'small convenience sample with no memory/bit-growth statistics. '
                        'No global speedup or SOTA conclusion; OpenSpiel not executed.'),
        'cases': [compare_case(gbt, name, a, repetitions) for name, a in games],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--seed', type=int, default=52052)
    parser.add_argument('--output', default='benchmark_gambit_v52_results.json')
    args = parser.parse_args(argv)
    result = run(repetitions=args.repetitions, seed=args.seed)
    # Only write an artifact after every fixture passed both exact certificates.
    with open(args.output, 'w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({
        'cases_checked': len(result['cases']),
        'all_exact_certificates_passed': all(c['both_exact_certificates_passed']
                                         for c in result['cases']),
        'pygambit_version': result['pygambit_version'],
        'fixtures_sha256': result['fixtures_sha256'],
        'result_file': args.output,
    }, indent=2))


if __name__ == '__main__':
    main()
