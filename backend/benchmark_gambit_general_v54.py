"""v54 independently validates full exact Nash families against Gambit enummixed.

Gambit returns extreme point profiles; GameTheory reports strict-support
rational H-polytopes. Point counts are never misrepresented as family counts.
"""
import argparse
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
from importlib.metadata import version
import json
import platform
import statistics
import sys
from time import perf_counter
import numpy as np
from general_exact_search import search_exact_nash_cells


def fixtures(seed=54054):
    rng = np.random.default_rng(seed)
    games = [
        ('battle_of_sexes', [[2, 0], [0, 1]], [[1, 0], [0, 2]]),
        ('prisoners_dilemma', [[3, 0], [5, 1]], [[3, 5], [0, 1]]),
        ('matching_pennies', [[1, -1], [-1, 1]], [[-1, 1], [1, -1]]),
        ('coordination_3x3', [[3, 0, 0], [0, 2, 0], [0, 0, 1]],
         [[3, 0, 0], [0, 2, 0], [0, 0, 1]]),
        ('all_ties_2x2', [[0, 0], [0, 0]], [[0, 0], [0, 0]]),
        ('all_ties_3x3', [[0]*3]*3, [[0]*3]*3),
        ('dominance_3x2', [[5, 4], [3, 2], [1, 0]], [[1, 1], [3, 2], [0, 4]]),
        ('continuous_line_3x3', [[0, 0, 0], [0, 0, 0], [-1, -1, -1]],
         [[3, 0, 0]]*3),
        ('rectangular_2x3', [[2, -2, 0], [-1, 3, 1]],
         [[-2, 1, 0], [4, -1, 2]]),
    ]
    for m, n, count in ((2, 2, 4), (2, 3, 2), (3, 2, 2),
                         (3, 3, 3), (4, 3, 1)):
        for j in range(count):
            games.append((f'seeded_{m}x{n}_{j}',
                          rng.integers(-5, 6, size=(m, n)),
                          rng.integers(-5, 6, size=(m, n))))
    return [(name, np.asarray(a, dtype=np.int64), np.asarray(bb, dtype=np.int64))
            for name, a, bb in games]


def fixture_digest(games):
    serial = [{'name': name, 'row': a.tolist(), 'col': b.tolist()}
              for name, a, b in games]
    return sha256(json.dumps(serial, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def nfg_bytes(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if (a.ndim != 2 or a.size == 0 or b.shape != a.shape or
            not np.issubdtype(a.dtype, np.integer) or
            not np.issubdtype(b.dtype, np.integer)):
        raise ValueError('Matching nonempty integer payoff matrices required')
    m, n = a.shape
    # NFG 1 R orders outcome records with row actions changing fastest.
    body = ' '.join(f'{int(a[i, j])} {int(b[i, j])}'
                    for j in range(n) for i in range(m))
    return (f'NFG 1 R "GameTheory v54" {{ "Row" "Column" }} '
            f'{{ {m} {n} }}\n\n{body}\n').encode('ascii')


def independent_exact_nash(a, b, x, y):
    """Original payoff best-response check, not GameTheory's Nash code."""
    a, b = np.asarray(a), np.asarray(b)
    if (a.ndim != 2 or a.size == 0 or b.shape != a.shape or
            len(x) != a.shape[0] or len(y) != a.shape[1]):
        raise AssertionError('Invalid Nash profile dimensions')
    x, y = [Fraction(v) for v in x], [Fraction(v) for v in y]
    if sum(x) != 1 or sum(y) != 1 or min(x+y) < 0:
        raise AssertionError('Nash profile not a probability simplex point')
    m, n = a.shape
    row_values = [sum(int(a[i, j])*y[j] for j in range(n))
                  for i in range(m)]
    col_values = [sum(x[i]*int(b[i, j]) for i in range(m))
                  for j in range(n)]
    best_row, best_col = max(row_values), max(col_values)
    actual_row = sum(x[i]*row_values[i] for i in range(m))
    actual_col = sum(y[j]*col_values[j] for j in range(n))
    if actual_row != best_row or actual_col != best_col:
        raise AssertionError('Exact Nash unilateral-deviation certificate failed')
    return actual_row, actual_col


def in_h_polytope(poly, strategy):
    indices = poly['variable_actions']
    active = [strategy[j] for j in indices]
    if (not indices or any(strategy[i] != 0 for i in range(len(strategy))
                           if i not in indices) or
            sum(active) != 1 or any(q <= 0 for q in active)):
        return False
    for eq in poly['equalities_dot_coefficients_eq_zero']:
        if sum(Fraction(c)*p for c, p in
               zip(eq['coefficients_exact'], active)) != 0:
            return False
    for inequality in poly['inequalities_dot_coefficients_ge_zero']:
        if sum(Fraction(c)*p for c, p in
               zip(inequality['coefficients_exact'], active)) < 0:
            return False
    return True


def containing_families(families, x, y):
    return [f for f in families if
            in_h_polytope(f['row_strategy_polytope'], x) and
            in_h_polytope(f['col_strategy_polytope'], y)]


def independently_pure_supports(a, b):
    m, n = a.shape
    return {(i, j) for i in range(m) for j in range(n)
            if int(a[i, j]) == max(int(a[k, j]) for k in range(m)) and
            int(b[i, j]) == max(int(b[i, k]) for k in range(n))}


def certified_reference(gbt, game, a, b, families):
    solution = gbt.nash.enummixed_solve(game, rational=True)
    if not solution.success or not solution.equilibria:
        raise AssertionError('Gambit failed to enumerate exact extreme equilibria')
    players = list(game.players)
    if len(players) != 2:
        raise AssertionError('Gambit changed number of players')
    row_labels = list(game.get_strategies(players[0]))
    col_labels = list(game.get_strategies(players[1]))
    gambit_supports = set()
    profiles = []
    for profile in solution.equilibria:
        x = [Fraction(str(profile[players[0]][s])) for s in row_labels]
        y = [Fraction(str(profile[players[1]][s])) for s in col_labels]
        row_value, col_value = independent_exact_nash(a, b, x, y)
        matching = containing_families(families, x, y)
        if len(matching) != 1:
            raise AssertionError('A Gambit exact equilibrium is absent or duplicated in GameTheory family union')
        support = (tuple(i for i, p in enumerate(x) if p > 0),
                   tuple(i for i, p in enumerate(y) if p > 0))
        expected = matching[0]['support']
        if support != (tuple(expected['row']), tuple(expected['col'])):
            raise AssertionError('Gambit support / GameTheory family mismatch')
        gambit_supports.add(support)
        profiles.append({'support': [list(support[0]), list(support[1])],
                         'row_value': str(row_value),
                         'col_value': str(col_value)})
    return profiles, gambit_supports


def compare_case(gbt, name, a, b, repetitions=2):
    if type(repetitions) is not int or not 1 <= repetitions <= 20:
        raise ValueError('repetitions must be integer 1..20')
    a, b = np.asarray(a), np.asarray(b)
    if (a.shape != b.shape or a.size > 16 or
            not np.issubdtype(a.dtype, np.integer) or
            not np.issubdtype(b.dtype, np.integer)):
        raise ValueError('Comparator supports matching integer games up to 4x4')
    m, n = a.shape
    full = (2**m-1)*(2**n-1)
    rational_times, gambit_times = [], []
    final = None
    for _ in range(repetitions):
        start = perf_counter()
        exact = search_exact_nash_cells(
            a, b, max_pairs=full, max_support=max(m,n),
            max_operations=2000000, feasibility_backend='simplex',
            prune_supports=True, mixed_pruning=False)
        rational_times.append(perf_counter()-start)
        if not exact['complete_equilibrium_set'] or exact['support_pairs_checked'] != full:
            raise AssertionError('Exact search was not exhaustive; completeness comparison invalid')
        families = exact['families']
        for f in families:
            x = [Fraction(t) for t in f['row_probabilities_exact']]
            y = [Fraction(t) for t in f['col_probabilities_exact']]
            independent_exact_nash(a, b, x, y)
            if len(containing_families(families, x, y)) != 1:
                raise AssertionError('GameTheory witness does not belong to exactly one family')
        pure_found = {(f['support']['row'][0], f['support']['col'][0])
                      for f in families if len(f['support']['row']) == 1
                      and len(f['support']['col']) == 1}
        actual_pure = independently_pure_supports(a, b)
        if pure_found != actual_pure:
            raise AssertionError('GameTheory pure support enumeration differs from independent direct scan')

        start = perf_counter()
        game = gbt.read_nfg(BytesIO(nfg_bytes(a, b)))
        recorded = game.to_arrays()
        if (len(recorded) != 2 or
                any(Fraction(str(recorded[0][i, j])) != int(a[i, j])
                    or Fraction(str(recorded[1][i, j])) != int(b[i, j])
                    for i in range(m) for j in range(n))):
            raise AssertionError('Gambit changed matrix entries or transpose')
        points, supports = certified_reference(gbt, game, a, b, families)
        gambit_times.append(perf_counter()-start)
        if not actual_pure.issubset({(p[0][0], p[1][0]) for p in supports
                                    if len(p[0]) == len(p[1]) == 1}):
            raise AssertionError('Gambit enumeration omitted a pure Nash equilibrium')
        if final is not None and final['gambit_extreme_count'] != len(points):
            raise AssertionError('Gambit enumeration count changed between repeats')
        final = {
            'case': name, 'shape': [m, n],
            'complete_exact_support_cells': len(families),
            'support_pairs_exhausted': full,
            'exact_lp_attempts': exact['lp_pairs_attempted'],
            'exact_operations': exact['exact_operations'],
            'gambit_extreme_count': len(points),
            'distinct_gambit_supports': len(supports),
            'pure_nash_count': len(actual_pure),
            'reference_profiles': points,
            'all_reference_points_exactly_certified_and_contained': True,
            'median_exact_seconds': None, 'median_gambit_seconds': None,
        }
    final['exact_all_seconds'] = rational_times
    final['gambit_all_seconds'] = gambit_times
    final['median_exact_seconds'] = statistics.median(rational_times)
    final['median_gambit_seconds'] = statistics.median(gambit_times)
    final['sampling_repetitions'] = repetitions
    return final


def run(repetitions=3, seed=54054):
    if type(repetitions) is not int or not 1 <= repetitions <= 20:
        raise ValueError('repetitions must be integer 1..20')
    import pygambit as gbt
    games = fixtures(seed)
    results = [compare_case(gbt, name, a, b, repetitions) for name, a, b in games]
    return {
        'release': 'v54', 'method': 'Gambit rational enummixed vs complete exact H-polytopes',
        'evidence': 'executed_independent_exact_solver',
        'seed': seed, 'repetitions': repetitions,
        'fixture_sha256': fixture_digest(games),
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'numpy': np.__version__, 'pygambit': version('pygambit'),
        'certification': 'exact best-response predicates and unique strict-support H-polytope membership',
        'scope': 'all reported Gambit extreme points contained, pure NE sets equal; NOT an equality proof of global equilibrium sets',
        'limitations': 'Gambit extreme point count is not GameTheory family count; short nonuniform paired samples cannot prove speed superiority or global SOTA.',
        'cases': results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--seed', type=int, default=54054)
    parser.add_argument('--output', default='benchmark_gambit_general_v54_results.json')
    args = parser.parse_args(argv)
    result = run(args.repetitions, args.seed)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({
        'cases_checked': len(result['cases']),
        'all_reference_profiles_certified': all(
            c['all_reference_points_exactly_certified_and_contained'] for c in result['cases']),
        'total_extreme_profiles': sum(c['gambit_extreme_count'] for c in result['cases']),
        'fixture_sha256': result['fixture_sha256'],
        'output': args.output,
    }, indent=2))


if __name__ == '__main__':
    main()
