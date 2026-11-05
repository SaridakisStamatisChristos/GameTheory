"""Bounded exact pure Nash and correlated equilibria for 3–5-player normal-form games.

Input is one payoff tensor per player with identical action dimensions.
All 'exact' claims apply to decimal rational values of validated float64
payoffs, not arbitrary-precision JSON tokens.
"""
from fractions import Fraction as F
from itertools import product
from math import prod
import numpy as np
from exact_rational_lp import _RationalSimplex


def _game(payoffs):
    try:
        data = np.asarray(payoffs, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Expected numeric rectangular player payoff tensors') from exc
    if data.ndim < 2 or not 3 <= data.shape[0] <= 5:
        raise ValueError('Expected 3 to 5 players with one payoff tensor per player')
    players = data.shape[0]
    if (data.ndim != players + 1 or
            any(not 1 <= actions <= 4 for actions in data.shape[1:]) or
            prod(data.shape[1:]) > 36 or
            not np.isfinite(data).all() or np.max(np.abs(data)) > 1000000):
        raise ValueError('Expected matching finite tensors: 1–4 actions per player, at most 36 profiles, reward magnitude <=1e6')
    shape = tuple(data.shape[1:])
    profiles = list(product(*(range(k) for k in shape)))
    utilities = [[F(str(float(v))) for v in data[p].flat] for p in range(players)]
    return shape, profiles, utilities


def solve_exact_multiplayer(payoffs, *, mode='correlated', objective='welfare',
                            max_operations=200000):
    """Complete pure Nash scan; optionally one primal/dual-certified exact CE.

    mode='pure' is an exhaustive pure-strategy analysis, not a general
    mixed Nash equilibrium algorithm. mode='correlated' additionally
    computes one exact joint recommendation distribution and, for
    welfare, an independently checked matching dual objective.
    """
    if mode not in ('pure', 'correlated'):
        raise ValueError('Unknown multiplayer solver mode')
    if objective not in ('welfare', 'feasible'):
        raise ValueError('Unknown multiplayer correlated objective')
    if type(max_operations) is not int or not 1 <= max_operations <= 2000000:
        raise ValueError('Invalid exact rational operation budget')
    shape, profiles, u = _game(payoffs)
    players, dimension = len(shape), len(profiles)
    index = {profile: k for k, profile in enumerate(profiles)}

    def deviated(profile, player, action):
        replaced = list(profile)
        replaced[player] = action
        return index[tuple(replaced)]

    pure = []
    for k, profile in enumerate(profiles):
        if all(all(u[p][deviated(profile, p, alternative)] <= u[p][k]
                   for alternative in range(shape[p]))
               for p in range(players)):
            pure.append({
                'actions': list(profile),
                'payoffs_exact': [str(u[p][k]) for p in range(players)],
            })
    result = {
        'players': players,
        'action_counts': list(shape),
        'profiles_checked': dimension,
        'pure_nash_equilibria': pure,
        'pure_nash_complete': True,
        'correlated_equilibrium_computed': False,
        'mixed_nash_complete': False,
        'input_model': 'decimal strings of validated float64 payoffs',
    }
    if mode == 'pure':
        result['status'] = 'certified_complete_pure_nash'
        return result

    constraints = [[F(1)]*dimension, [-F(1)]*dimension]
    rhs = [F(1), -F(1)]
    deviation_order = []
    for p, actions in enumerate(shape):
        for recommended in range(actions):
            for alternative in range(actions):
                if alternative == recommended:
                    continue
                row = [F(0)]*dimension
                for k, profile in enumerate(profiles):
                    if profile[p] == recommended:
                        row[k] = u[p][deviated(profile, p, alternative)] - u[p][k]
                constraints.append(row)
                rhs.append(F(0))
                deviation_order.append([p, recommended, alternative])

    welfare = [sum(u[p][k] for p in range(players))
               for k in range(dimension)]
    objective_vector = welfare if objective == 'welfare' else [F(0)]*dimension
    work = {'steps': 0, 'limit': max_operations}
    joint = _RationalSimplex(constraints, rhs, objective_vector, work).solve()
    if (joint is None or len(joint) != dimension or
            any(q < 0 for q in joint) or sum(joint) != 1):
        raise AssertionError('Exact multiplayer CE probability simplex verification failed')

    # An independent pure-deviation audit does not reuse LP constraint rows.
    deviation_gains = []
    for p, recommended, alternative in deviation_order:
        gain = sum(joint[k] * (u[p][deviated(profile, p, alternative)] - u[p][k])
                   for k, profile in enumerate(profiles)
                   if profile[p] == recommended)
        deviation_gains.append(gain)
    if any(gain > 0 for gain in deviation_gains):
        raise AssertionError('Exact multiplayer CE recommendation obedience failed')

    # For max c*q under A*q<=b and q>=0, the exact dual is min b*y
    # with A.T*y>=c, y>=0. Solve it independently and recheck all rows.
    if objective == 'welfare':
        dual = _RationalSimplex(
            [[-constraints[i][j] for i in range(len(rhs))]
             for j in range(dimension)],
            [-v for v in objective_vector], [-v for v in rhs], work).solve()
        if dual is None:
            raise AssertionError('Exact multiplayer CE dual LP infeasible')
    else:
        dual = [F(0)]*len(rhs)
    optimum = sum(joint[k]*objective_vector[k] for k in range(dimension))
    if (len(dual) != len(rhs) or any(v < 0 for v in dual) or
            any(sum(constraints[i][j]*dual[i] for i in range(len(rhs)))
                < objective_vector[j] for j in range(dimension)) or
            sum(rhs[i]*dual[i] for i in range(len(rhs))) != optimum):
        raise AssertionError('Exact multiplayer CE primal-dual optimality failed')

    values = [sum(joint[k]*u[p][k] for k in range(dimension))
              for p in range(players)]
    result.update({
        'status': 'certified_exact_multiplayer_correlated',
        'correlated_equilibrium_computed': True,
        'correlated_equilibrium_set_complete': False,
        'action_profiles': [list(profile) for profile in profiles],
        'joint_distribution_exact': [str(q) for q in joint],
        'expected_payoffs_exact': [str(v) for v in values],
        'obedience_inequalities_checked': len(deviation_order),
        'maximum_obedience_gain_exact': str(max([F(0), *deviation_gains])),
        'objective': objective,
        'objective_value_exact': str(optimum),
        'dual_multipliers_exact': [str(v) for v in dual],
        'dual_constraint_order': 'normalization upper, normalization lower, then (player, recommended action, alternative action) in ascending order',
        'exact_operations': work['steps'],
        'certificate': 'exact joint simplex and independent unconditional recommendation-obedience checks',
        'optimality_evidence': 'exact rational primal/dual feasibility and equality of objectives',
    })
    return result
