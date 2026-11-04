"""Exact zero-sum minimax strategies from two bounded rational LPs.

The strictly-positive payoff translation permits two nonnegative standard-form
LPs. Primal/dual equality and *original-payoff* security constraints are
verified in exact rational arithmetic independently of the simplex tables.
"""
from fractions import Fraction
import numpy as np
from exact_rational_lp import _RationalSimplex


def solve_exact_zero_sum(payoffs, *, max_operations=200000):
    """Return one exact two-player zero-sum saddle point, not its entire set.

    Payoffs are interpreted as decimal strings of validated float64 entries.
    Operation budget limits pivot tableau updates, not fraction bit-growth.
    """
    try:
        a = np.asarray(payoffs, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Expected a rectangular numeric payoff matrix') from exc
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError('Expected a nonempty finite payoff matrix')
    if a.size > 100:
        raise ValueError('Exact minimax LP limited to 100 action profiles')
    if type(max_operations) is not int or not 1 <= max_operations <= 2000000:
        raise ValueError('Invalid exact minimax work budget')

    m, n = a.shape
    original = [[Fraction(str(v)) for v in row] for row in a]
    shift = Fraction(1) - min(v for row in original for v in row)
    positive = [[value + shift for value in row] for row in original]
    work = {'steps': 0, 'limit': max_operations}

    # Row security: min sum(u) such that M.T @ u >= 1.
    # Column security: max sum(v) such that M @ v <= 1.
    # For M > 0 the two LPs are feasible/bounded and are exact duals.
    row_scaled = _RationalSimplex(
        [[-positive[i][j] for i in range(m)] for j in range(n)],
        [-Fraction(1)]*n, [-Fraction(1)]*m, work).solve()
    col_scaled = _RationalSimplex(
        positive, [Fraction(1)]*m, [Fraction(1)]*n, work).solve()
    if row_scaled is None or col_scaled is None:
        raise AssertionError('Positive zero-sum minimax LP unexpectedly infeasible')
    if (len(row_scaled) != m or len(col_scaled) != n or
            any(v < 0 for v in (*row_scaled, *col_scaled))):
        raise AssertionError('Exact minimax simplex returned invalid scaled strategies')
    row_mass, col_mass = sum(row_scaled), sum(col_scaled)
    if row_mass <= 0 or col_mass <= 0 or row_mass != col_mass:
        raise AssertionError('Exact minimax primal-dual objective equality failed')

    row = [v / row_mass for v in row_scaled]
    col = [v / col_mass for v in col_scaled]
    value = Fraction(1) / row_mass - shift
    row_guarantee = min(sum(row[i]*original[i][j] for i in range(m))
                        for j in range(n))
    col_ceiling = max(sum(original[i][j]*col[j] for j in range(n))
                      for i in range(m))
    expected = sum(row[i]*original[i][j]*col[j]
                   for i in range(m) for j in range(n))
    if (sum(row) != 1 or sum(col) != 1 or
            any(v < 0 for v in (*row, *col)) or
            row_guarantee != value or col_ceiling != value or expected != value):
        raise AssertionError('Exact minimax original-payoff Nash certificate failed')

    return {
        'row_strategy_exact': [str(v) for v in row],
        'col_strategy_exact': [str(v) for v in col],
        'value_exact': str(value),
        'row_security_value_exact': str(row_guarantee),
        'col_security_value_exact': str(col_ceiling),
        'minimax_duality_gap_exact': '0',
        'lp_objective_exact': str(row_mass),
        'positive_payoff_shift_exact': str(shift),
        'exact_operations': work['steps'],
        'certificate': 'exact rational dual LP equality and original-payoff saddle-point inequalities',
        'status': 'certified_exact_minimax',
        'all_equilibria_enumerated': False,
        'input_model': 'decimal strings of validated float64 payoffs',
    }
