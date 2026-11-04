"""Bounded exact-rational correlated equilibrium for finite bimatrix games.

Computes an LP-selected joint distribution and independently checks every
unconditional recommendation-obedience inequality in rational arithmetic.
"""
from fractions import Fraction
import numpy as np
from exact_rational_lp import _RationalSimplex


def solve_exact_correlated_equilibrium(a, b, *, objective='welfare', max_operations=200000):
    try:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Payoffs must be numeric rectangular matrices') from exc
    if (a.ndim != 2 or a.size == 0 or b.shape != a.shape or
            not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Expected nonempty, finite, matching payoff matrices')
    if a.size > 36:
        raise ValueError('Exact correlated-equilibrium LP limited to 36 action profiles')
    if objective not in ('welfare', 'feasible'):
        raise ValueError('Invalid correlated-equilibrium objective')
    if type(max_operations) is not int or not 1 <= max_operations <= 2000000:
        raise ValueError('Invalid rational-work budget')
    m, n = a.shape
    af = [[Fraction(str(value)) for value in row] for row in a]
    bf = [[Fraction(str(value)) for value in row] for row in b]
    dim = m*n
    constraints = [[Fraction(1)]*dim, [Fraction(-1)]*dim]
    rhs = [Fraction(1), Fraction(-1)]
    for i in range(m):
        for k in range(m):
            if i != k:
                row = [Fraction(0)]*dim
                for j in range(n):
                    row[i*n+j] = af[k][j]-af[i][j]
                constraints.append(row)
                rhs.append(Fraction(0))
    for j in range(n):
        for k in range(n):
            if j != k:
                row = [Fraction(0)]*dim
                for i in range(m):
                    row[i*n+j] = bf[i][k]-bf[i][j]
                constraints.append(row)
                rhs.append(Fraction(0))
    objective_vector = [af[i][j]+bf[i][j] if objective == 'welfare'
                        else Fraction(0) for i in range(m) for j in range(n)]
    work = {'steps': 0, 'limit': max_operations}
    flat = _RationalSimplex(constraints, rhs, objective_vector, work).solve()
    if flat is None:
        raise AssertionError('Finite bimatrix game unexpectedly has no correlated equilibrium')
    if sum(flat) != 1 or any(p < 0 for p in flat):
        raise AssertionError('Exact LP failed probability-simplex verification')
    q = [flat[i*n:(i+1)*n] for i in range(m)]
    row_gains = [sum(q[i][j]*(af[k][j]-af[i][j]) for j in range(n))
                 for i in range(m) for k in range(m) if i != k]
    col_gains = [sum(q[i][j]*(bf[i][k]-bf[i][j]) for i in range(m))
                 for j in range(n) for k in range(n) if j != k]
    if any(g > 0 for g in (*row_gains, *col_gains)):
        raise AssertionError('Exact LP failed obedience verification')
    if objective == 'welfare':
        # Dual of max c*x with A*x<=b, x>=0: min b*y, A.T*y>=c, y>=0.
        # Solve it separately, then check weak-duality equality in exact arithmetic.
        dual = _RationalSimplex(
            [[-constraints[i][j] for i in range(len(rhs))] for j in range(dim)],
            [-v for v in objective_vector], [-v for v in rhs], work).solve()
        if dual is None:
            raise AssertionError('Exact correlated-equilibrium dual LP unexpectedly infeasible')
    else:
        dual = [Fraction(0)]*len(rhs)
    if (len(dual) != len(rhs) or any(v < 0 for v in dual) or
            any(sum(constraints[i][j]*dual[i] for i in range(len(rhs))) < objective_vector[j]
                for j in range(dim)) or
            sum(dual[i]*rhs[i] for i in range(len(rhs))) !=
            sum(flat[j]*objective_vector[j] for j in range(dim))):
        raise AssertionError('Exact correlated-equilibrium dual optimality certificate failed')
    values = [sum(q[i][j]*payoff[i][j] for i in range(m) for j in range(n))
              for payoff in (af, bf)]
    return {
        'joint_distribution_exact': [[str(p) for p in row] for row in q],
        'expected_payoffs_exact': [str(v) for v in values],
        'maximum_obedience_violation_exact': '0',
        'obedience_inequalities_checked': len(row_gains)+len(col_gains),
        'objective': objective,
        'objective_value_exact': str(sum(values)) if objective == 'welfare' else '0',
        'exact_operations': work['steps'],
        'certificate': 'independent exact rational probability and unconditional obedience checks',
        'dual_multipliers_exact': [str(v) for v in dual],
        'dual_constraint_order': 'normalization upper, normalization lower, row deviations (i,k) with i!=k, then column deviations (j,k) with j!=k',
        'optimality_evidence': 'exact rational primal and dual feasibility with matching objective values',
        'input_model': 'decimal strings of validated float64 payoffs',
        'complete_equilibrium_set': False,
    }
