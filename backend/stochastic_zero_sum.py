"""Bounded Shapley value iteration for discounted finite zero-sum Markov games.

The local zero-sum matrix saddle is obtained from independent primal/dual LPs.
A posteriori security intervals bound the game value and stationary-policy
regret under the declared floating-point input/solver model. These are not
formal interval-arithmetic certificates or arbitrary-precision results.
"""
import numpy as np
from scipy.optimize import linprog


def _validate_model(rewards, transitions, discount, tolerance, max_iterations):
    try:
        rewards = np.asarray(rewards, dtype=float)
        transitions = np.asarray(transitions, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Rewards and transitions must be numeric rectangular arrays') from exc
    if rewards.ndim != 3 or not rewards.size or not np.isfinite(rewards).all():
        raise ValueError('Expected finite rewards with shape [states, row_actions, col_actions]')
    states, rows, cols = rewards.shape
    if (states > 8 or rows > 6 or cols > 6 or
            transitions.shape != (states, rows, cols, states) or
            not np.isfinite(transitions).all()):
        raise ValueError('Transition shape must be [states, row_actions, col_actions, states]; limits 8 states and 6x6 actions')
    if (np.any(transitions < 0) or
            np.any(np.abs(transitions.sum(axis=-1)-1) > 1e-12)):
        raise ValueError('Transition probabilities must be nonnegative and each row sum to one')
    if np.max(np.abs(rewards)) > 1e6:
        raise ValueError('Absolute immediate reward exceeds 1e6')
    if (type(discount) not in (float, int) or not np.isfinite(discount)
            or discount < 0 or discount >= 1):
        raise ValueError('Discount must be a finite real in [0,1)')
    if (type(tolerance) not in (float, int) or not np.isfinite(tolerance)
            or not 1e-10 <= tolerance <= 1):
        raise ValueError('Tolerance must be finite and between 1e-10 and 1')
    if (type(max_iterations) is not int or not 1 <= max_iterations <= 500 or
            states * max_iterations > 4000):
        raise ValueError('Invalid iteration and state work budget')
    # Re-normalizing tolerated rounding errors preserves a stochastic operator.
    sums = transitions.sum(axis=-1, keepdims=True)
    normalization_delta = float(np.max(np.abs(sums - 1)))
    return rewards, transitions / sums, float(discount), float(tolerance), normalization_delta


def _matrix_saddle(q):
    """Return feasible saddle policies with independently recomputed payoff bounds."""
    m, n = q.shape
    if m == 1 and n == 1:
        return np.ones(1), np.ones(1), float(q[0, 0]), float(q[0, 0])
    lo, hi = float(np.min(q)), float(np.max(q))
    if lo == hi:
        x = np.zeros(m); y = np.zeros(n)
        x[0] = 1.; y[0] = 1.
        return x, y, lo, lo
    center = lo + (hi-lo)/2
    scale = max(float(np.max(np.abs(q-center))), np.finfo(float).tiny)
    normal = (q-center)/scale
    row = linprog(np.r_[np.zeros(m), -1.],
                  A_ub=np.c_[-normal.T, np.ones(n)], b_ub=np.zeros(n),
                  A_eq=np.array([[*([1.]*m), 0.]]), b_eq=[1.],
                  bounds=[(0,None)]*m+[(None,None)], method='highs')
    col = linprog(np.r_[np.zeros(n), 1.],
                  A_ub=np.c_[normal, -np.ones(m)], b_ub=np.zeros(m),
                  A_eq=np.array([[*([1.]*n), 0.]]), b_eq=[1.],
                  bounds=[(0,None)]*n+[(None,None)], method='highs')
    if not (row.success and col.success):
        raise ValueError('Stage minimax LP was numerically inconclusive')
    x, y = np.maximum(row.x[:m], 0), np.maximum(col.x[:n], 0)
    if (x.sum() <= 0 or y.sum() <= 0 or
            not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Stage minimax LP returned invalid probabilities')
    x /= x.sum(); y /= y.sum()
    gap = float(np.max(normal @ y)-np.min(x @ normal))
    if gap > 1e-7 or gap < -1e-7:
        raise ValueError('Stage minimax normalized saddle certificate failed')
    extended_q = q.astype(np.longdouble)
    lower = float(np.min(x.astype(np.longdouble) @ extended_q))
    upper = float(np.max(extended_q @ y.astype(np.longdouble)))
    if upper + 1e-10*max(1., abs(lower), abs(upper)) < lower:
        raise ValueError('Stage security bounds inconsistent')
    return x, y, lower, upper


def solve_discounted_zero_sum_stochastic(rewards, transitions, *, discount=0.9,
                                          tolerance=1e-6, max_iterations=200):
    """Report an a posteriori bound, or explicitly label an unfinished solve.

    Returns 'converged_numerical' only when the value-error estimate
    from the security intervals is <= tolerance. The 'partial_bound'
    result never silently claims completion.
    """
    r, p, gamma, tol, normalization_delta = _validate_model(
        rewards, transitions, discount, tolerance, max_iterations)
    states, rows, cols = r.shape
    v = np.zeros(states)
    stage_lp_solves = 0

    def stage_pass(values):
        q = r + gamma * np.einsum('aijb,b->aij', p, values)
        xpolicy = []; ypolicy = []; lower = []; upper = []
        for s in range(states):
            x, y, lo, hi = _matrix_saddle(q[s])
            xpolicy.append(x.tolist()); ypolicy.append(y.tolist())
            lower.append(lo); upper.append(hi)
        return np.asarray(lower), np.asarray(upper), xpolicy, ypolicy

    for iteration in range(max_iterations+1):
        lower, upper, xpolicy, ypolicy = stage_pass(v)
        stage_lp_solves += states
        # Actual Shapley operator lies in the stage security intervals.
        rho = max(float(np.max(np.abs(lower-v))),
                  float(np.max(np.abs(upper-v))))
        value_error = rho/(1-gamma)
        if not np.isfinite(value_error):
            raise ValueError('Residual bound overflowed; choose safer discount/reward magnitudes')
        if value_error <= tol or iteration == max_iterations:
            return {
                'status': 'converged_numerical' if value_error <= tol else 'partial_bound',
                'tolerance_met': bool(value_error <= tol),
                'states': states,
                'actions': {'row': rows, 'column': cols},
                'discount': gamma,
                'iterations': iteration,
                'stage_games_solved': stage_lp_solves,
                'value_estimates': v.tolist(),
                'row_stationary_policy': xpolicy,
                'col_stationary_policy': ypolicy,
                'row_security_values': lower.tolist(),
                'column_ceiling_values': upper.tolist(),
                'state_saddle_gaps': np.maximum(0, upper-lower).tolist(),
                'max_bellman_residual_bound': rho,
                'value_error_bound': value_error,
                'stationary_policy_regret_bound': 2*value_error,
                'requested_value_error_tolerance': tol,
                'transition_max_row_sum_correction': normalization_delta,
                'certificate': 'numerical saddle security intervals plus discounted Shapley contraction; floating-point, not formal interval arithmetic',
                'note': 'Stationary policies are candidates at the reported value iterate. Bound is conditional on the numerical input and LP certificates; partial status never asserts convergence.',
            }
        v = (lower+upper)/2
