"""Independent floating-point Lemke–Howson complementary pivot implementation.

No Nashpy imports. Lexicographic ratio ties use symbolic RHS perturbations
represented by the original slack basis inverse. Numerical certificates
are computed on the original, unshifted payoff matrices.
"""
import numpy as np


def _tableau(coefs, slack_first):
    r, c = coefs.shape
    mat = np.column_stack((np.eye(r), coefs)) if slack_first else np.column_stack((coefs, np.eye(r)))
    basis = list(range(r)) if slack_first else list(range(c, c+r))
    slack = list(range(r)) if slack_first else list(range(c, c+r))
    return mat, basis, slack


def _pivot(mat, basis, slack, entering, *, eps=1e-10):
    # Recompute the current basic solution from the original constraints,
    # rather than accumulating tableau row-operation errors.
    inv = np.linalg.inv(mat[:, basis])
    direction = inv @ mat[:, entering]
    rhs = inv @ np.ones(mat.shape[0])
    eligible = np.flatnonzero(direction > eps)
    if not len(eligible):
        raise ArithmeticError('Unbounded complementary pivot ray')
    # Lexicographic minimum ratio on [rhs, infinitesimal slack perturbations].
    # Round only at tie resolution to reduce spurious near-ties.
    keys = [(float(rhs[i]/direction[i]), *[float(z) for z in (inv @ mat[:, slack])[i]/direction[i]], int(i)) for i in eligible]
    minimum = min(k[0] for k in keys)
    tied = [k for k in keys if abs(k[0]-minimum) <= eps*max(1., abs(minimum))]
    row = min(tied, key=lambda k: (*k[1:-1], k[-1]))[-1]
    leaving = basis[row]
    basis[row] = entering
    return leaving


def solve_independent_lh(a, b, *, tol=1e-7, max_labels=None, max_pivots=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or not a.size or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected equally shaped nonempty finite payoff matrices')
    m, n = a.shape
    if max_labels is None: max_labels = m+n
    if not isinstance(max_labels,int) or not 1 <= max_labels <= m+n or max_pivots < 1:
        raise ValueError('Invalid label or pivot budget')
    aa = a - min(0.,float(a.min())) + 1.
    bb = b - min(0.,float(b.min())) + 1.
    # P: B.T x <= 1, labels 0..m-1 are row actions; m..m+n-1 slacks.
    pm, pb, ps = _tableau(bb.T, False)
    # Q: A y <= 1, labels 0..m-1 slacks; m..m+n-1 column actions.
    qm, qb, qs = _tableau(aa, True)
    results, failures = [], []
    for initial in range(max_labels):
        p_basis, q_basis = pb.copy(), qb.copy()
        entering = initial
        which = 'p' if entering not in p_basis else 'q'
        seen = set()
        try:
            for step in range(max_pivots):
                state = (which, entering, tuple(p_basis), tuple(q_basis))
                if state in seen: raise ArithmeticError('Complementary pivot cycle')
                seen.add(state)
                if which == 'p':
                    entering = _pivot(pm, p_basis, ps, entering)
                    which = 'q'
                else:
                    entering = _pivot(qm, q_basis, qs, entering)
                    which = 'p'
                if entering == initial: break
            else: raise ArithmeticError('Pivot budget exhausted')
            px = np.linalg.solve(pm[:,p_basis], np.ones(n))
            qy = np.linalg.solve(qm[:,q_basis], np.ones(m))
            x = np.zeros(m); y = np.zeros(n)
            for i,label in enumerate(p_basis):
                if label < m: x[label] = px[i]
            for i,label in enumerate(q_basis):
                if label >= m: y[label-m] = qy[i]
            if x.sum() <= 0 or y.sum() <= 0: raise ArithmeticError('Zero strategy mass')
            x /= x.sum(); y /= y.sum()
            if min(x.min(),y.min()) < -tol: raise ArithmeticError('Negative strategy mass')
            gap = max(0.,float(np.max(a@y)-x@a@y),float(np.max(x@b)-x@b@y))
            if gap > tol: raise ArithmeticError(f'Nash gap {gap:.6g} exceeds tolerance')
            if not any(np.max(abs(x-np.array(e['row_probabilities']))) <= tol and np.max(abs(y-np.array(e['col_probabilities']))) <= tol for e in results):
                results.append({'row_probabilities':x.tolist(),'col_probabilities':y.tolist(), 'expected_payoffs':[float(x@a@y),float(x@b@y)], 'nash_gap':gap,'initial_label':initial,'pivots':step+1})
        except (ArithmeticError,ValueError,np.linalg.LinAlgError,OverflowError) as exc:
            failures.append({'initial_label':initial,'error':str(exc)})
    return {'method':'independent_lemke_howson','equilibria':results,'failures':failures,'labels_attempted':max_labels,'complete':False}
