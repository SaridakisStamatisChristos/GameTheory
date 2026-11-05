"""Bounded N-player mixed-Nash candidate search with exact rational epsilon certificates.

Support indifference equations are nonlinear/multilinear for N >= 3.
Numerical root discovery is incomplete. Every returned candidate is independently
checked as an epsilon-Nash profile over exact decimal rational payoffs and
probabilities; zero epsilon alone permits a claim of exact rational Nash.
"""
from fractions import Fraction as F
from itertools import combinations, product
from math import prod

import numpy as np
from scipy.optimize import root
from multiplayer_exact import _game, solve_exact_multiplayer


class _EvaluationBudget(Exception):
    """Internal signal to terminate numerical search without claiming completion."""


def solve_multiplayer_mixed(payoffs, *, tolerance=1e-8, seed=59059,
                            max_support_patterns=96, starts_per_pattern=4,
                            max_evaluations=10000, max_candidates=16):
    """Return exact-rational epsilon-verified *candidates*, not all mixed Nash.

    All pure equilibria are fully enumerated independently. Mixed-support roots
    are sought with deterministic initial guesses and seeded Dirichlet restarts.
    The search can miss equilibria even after every pattern has been visited.
    """
    if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or not 1e-12 <= tolerance <= 0.1:
        raise ValueError('tolerance must be finite and between 1e-12 and 0.1')
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError('seed must be a nonnegative bounded integer')
    if type(max_support_patterns) is not int or not 1 <= max_support_patterns <= 1024:
        raise ValueError('max_support_patterns must be integer in 1..1024')
    if type(starts_per_pattern) is not int or not 1 <= starts_per_pattern <= 12:
        raise ValueError('starts_per_pattern must be integer in 1..12')
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= 100000:
        raise ValueError('max_evaluations must be integer in 1..100000')
    if type(max_candidates) is not int or not 1 <= max_candidates <= 128:
        raise ValueError('max_candidates must be integer in 1..128')

    shape, profiles, utilities = _game(payoffs)
    players, dimension = len(shape), len(profiles)
    profile_indices = np.asarray(profiles, dtype=int)
    numeric_payoffs = np.asarray([[float(v) for v in player] for player in utilities])
    pure = solve_exact_multiplayer(payoffs, mode='pure')
    action_supports = [
        tuple(combinations(range(actions), size)
              for size in range(1, actions+1))
        for actions in shape
    ]
    # Flatten the size-grouped generators without relying on sorted dictionary keys.
    possible = [tuple(s for group in groups for s in group)
                for groups in action_supports]
    patterns = sorted(product(*possible), key=lambda s: (sum(map(len, s)), s))
    total_patterns = prod(2**actions-1 for actions in shape)
    if len(patterns) != total_patterns:
        raise AssertionError('Mixed support pattern enumeration mismatch')

    rng = np.random.default_rng(seed)
    tolerance_rational = F(str(tolerance))
    patterns_done = 0
    numerical_calls = 0
    evaluations = 0
    accepted = []
    seen = set()
    stop_reason = None

    def verify(probabilities, supports):
        # Construct rational numbers with sum=1 *exactly*, avoiding a float
        # normalization residual being mistaken for a strategic deviation.
        rational = []
        for player, (x, support) in enumerate(zip(probabilities, supports)):
            if (len(x) != shape[player] or
                    any(not np.isfinite(x[a]) for a in support) or
                    any(x[a] <= 1e-8 for a in support) or
                    any(x[a] != 0 for a in range(shape[player]) if a not in support)):
                return None
            vec = [F(0)]*shape[player]
            for a in support[:-1]:
                vec[a] = F(str(float(x[a])))
            vec[support[-1]] = F(1)-sum(vec)
            if any(vec[a] <= 0 for a in support) or sum(vec) != 1:
                return None
            rational.append(vec)

        regrets = []
        expected_values = []
        supported_spreads = []
        for p in range(players):
            by_action = [F(0)]*shape[p]
            for k, profile in enumerate(profiles):
                weight = prod(rational[q][profile[q]] for q in range(players) if q != p)
                by_action[profile[p]] += weight*utilities[p][k]
            expected = sum(rational[p][a]*by_action[a] for a in range(shape[p]))
            regret = max(by_action)-expected
            spread = max(by_action[a] for a in supports[p])-min(
                by_action[a] for a in supports[p])
            if regret < 0:
                raise AssertionError('Exact Nash regret cannot be negative')
            regrets.append(regret)
            expected_values.append(expected)
            supported_spreads.append(spread)
        epsilon = max(regrets)
        # Require both genuine epsilon-Nash and supported-action near-indifference,
        # which prevents tiny probabilities from disguising poor root solutions.
        if epsilon > tolerance_rational or max(supported_spreads) > 2*tolerance_rational:
            return None
        return {
            'support': [list(s) for s in supports],
            'strategies_exact': [[str(v) for v in player] for player in rational],
            'expected_payoffs_exact': [str(v) for v in expected_values],
            'player_unilateral_regrets_exact': [str(v) for v in regrets],
            'max_supported_indifference_spread_exact': str(max(supported_spreads)),
            'epsilon_exact': str(epsilon),
            'status': 'certified_exact_rational_nash' if epsilon == 0
                      else 'verified_exact_epsilon_nash',
        }

    for pattern in patterns:
        if patterns_done >= max_support_patterns:
            stop_reason = 'support_pattern_budget'
            break
        if len(accepted) >= max_candidates:
            stop_reason = 'candidate_budget'
            break
        if all(len(s) == 1 for s in pattern):
            patterns_done += 1
            continue

        dimensions = [len(s)-1 for s in pattern]

        def unpack(z):
            result=[]
            offset=0
            for p, (support, count) in enumerate(zip(pattern,dimensions)):
                x=np.zeros(shape[p])
                if count:
                    x[list(support[:-1])]=z[offset:offset+count]
                x[support[-1]]=1.-np.sum(z[offset:offset+count])
                result.append(x)
                offset+=count
            return result

        def equations(z):
            nonlocal evaluations
            if evaluations >= max_evaluations:
                raise _EvaluationBudget()
            evaluations += 1
            probabilities=unpack(z)
            result=[]
            for p, support in enumerate(pattern):
                if len(support)==1:
                    continue
                weights=np.ones(dimension)
                for other in range(players):
                    if other != p:
                        weights *= probabilities[other][profile_indices[:,other]]
                action_values=np.bincount(profile_indices[:,p],
                    weights=weights*numeric_payoffs[p],minlength=shape[p])
                result.extend(action_values[list(support[1:])]-action_values[support[0]])
            return np.asarray(result,dtype=float)

        for start in range(starts_per_pattern):
            if len(accepted) >= max_candidates:
                stop_reason='candidate_budget'
                break
            initial=[]
            for s in pattern:
                if len(s)>1:
                    probs = np.full(len(s), 1/len(s)) if start==0 else rng.dirichlet(
                        np.ones(len(s)))
                    initial.extend(probs[:-1])
            numerical_calls += 1
            try:
                solution = root(equations, initial, method='hybr',
                                options={'maxfev':180})
            except _EvaluationBudget:
                stop_reason='function_evaluation_budget'
                break
            if not np.isfinite(solution.x).all():
                continue
            probabilities=unpack(solution.x)
            candidate=verify(probabilities,pattern)
            if candidate is not None:
                # Dedupe nearby root basins while retaining exact rational
                # independent incentive audits in each returned candidate.
                key=(tuple(pattern),tuple(round(float(F(v)),8)
                     for player in candidate['strategies_exact'] for v in player))
                if key not in seen:
                    seen.add(key)
                    accepted.append(candidate)
        if stop_reason is not None:
            break
        patterns_done += 1

    all_patterns_visited = stop_reason is None and patterns_done == total_patterns
    return {
        'status': 'certified_candidates' if accepted else 'no_mixed_candidates_verified',
        'players': players,
        'action_counts': list(shape),
        'profiles_checked_for_pure_nash': dimension,
        'pure_nash_equilibria': pure['pure_nash_equilibria'],
        'pure_nash_complete': True,
        'mixed_nash_complete': False,
        'mixed_candidates': accepted,
        'mixed_candidates_found': len(accepted),
        'support_patterns_total': total_patterns,
        'support_patterns_examined': patterns_done,
        'all_support_patterns_attempted': all_patterns_visited,
        'root_calls': numerical_calls,
        'function_evaluations': evaluations,
        'incomplete_reason': stop_reason,
        'requested_epsilon_tolerance_exact': str(tolerance_rational),
        'seed': seed,
        'certificate': 'each returned mixed profile independently rechecked using exact rational simplex probabilities and pure unilateral deviation payoffs',
        'limitation': 'numerical support roots are not exhaustive; full support-pattern traversal does not prove all mixed Nash equilibria found',
        'input_model': 'decimal strings of validated float64 payoffs',
    }
