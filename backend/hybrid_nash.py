"""Adaptive finite bimatrix equilibrium search with explicit coverage semantics."""
import numpy as np
from advanced import support_equilibria
from support_lp import solve_support_lp


def solve_hybrid(a, b, *, max_pairs=100000, fast_support=4, tol=1e-7, exhaustive=False):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or a.size == 0 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected equally shaped, nonempty finite payoff matrices')
    if max_pairs < 1 or fast_support < 1:
        raise ValueError('Budgets must be positive')
    m, n = a.shape
    pairs = (2**m-1)*(2**n-1)
    fast = support_equilibria(a, b, tol=tol, max_support=fast_support)
    for eq in fast:
        x = np.asarray(eq['row_probabilities']); y = np.asarray(eq['col_probabilities'])
        eq['nash_gap'] = max(0., float(np.max(a@y)-x@a@y), float(np.max(x@b)-x@b@y))
    # Identical payoff rows/columns for both players certify degeneracy risk.
    duplicate_rows = any(np.array_equal(a[i], a[j]) and np.array_equal(b[i], b[j]) for i in range(m) for j in range(i+1,m))
    duplicate_cols = any(np.array_equal(a[:,i], a[:,j]) and np.array_equal(b[:,i], b[:,j]) for i in range(n) for j in range(i+1,n))
    degeneracy_risk = duplicate_rows or duplicate_cols
    if fast and not exhaustive and not degeneracy_risk:
        return {'equilibria':fast,'method':'fast_support','complete_support_coverage':False,
                'support_pairs_checked':None,'support_pairs_total':pairs,
                'note':'Certified representative equilibria; other equilibria may exist.'}
    if pairs > max_pairs:
        if exhaustive:
            raise ValueError(f'{pairs} support pairs exceed max_pairs={max_pairs}')
        return {'equilibria':fast,'method':'fast_support_budget_limited','complete_support_coverage':False,
                'support_pairs_checked':None,'support_pairs_total':pairs,
                'note':'LP fallback skipped due to support-pair budget; absence of results is not a nonexistence certificate.'}
    lp = solve_support_lp(a, b, max_pairs=max_pairs, tol=min(tol/10, 1e-8))
    lp.update(method='support_lp', support_pairs_total=pairs)
    return lp
