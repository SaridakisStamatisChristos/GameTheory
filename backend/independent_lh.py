"""Independent floating-point Lemke–Howson complementary pivot implementation.

No Nashpy imports. Lexicographic ratio ties use symbolic RHS perturbations
represented by the original slack basis inverse. Numerical certificates
are computed on the original, unshifted payoff matrices.
"""
import numpy as np
from numerical_certificate import certify_bimatrix
from safe_numeric import expected_payoffs


def _tableau(coefs, slack_first):
    r, c = coefs.shape
    mat = np.column_stack((np.eye(r), coefs)) if slack_first else np.column_stack((coefs, np.eye(r)))
    basis = list(range(r)) if slack_first else list(range(c, c+r))
    slack = list(range(r)) if slack_first else list(range(c, c+r))
    return mat, basis, slack


def _pivot(mat, basis, slack, entering, *, eps=1e-10):
    # Recompute the current basic solution from the original constraints,
    # rather than accumulating tableau row-operation errors.
    basic = mat[:, basis]
    # One factorization/solve with multiple RHS, avoiding an explicit inverse.
    solved = np.linalg.solve(basic, np.column_stack((mat[:, entering], np.ones(mat.shape[0]), mat[:, slack])))
    direction = solved[:, 0]
    rhs = solved[:, 1]
    perturbation = solved[:, 2:]
    eligible = np.flatnonzero(direction > eps)
    if not len(eligible):
        raise ArithmeticError('Unbounded complementary pivot ray')
    # Lexicographic minimum ratio on [rhs, infinitesimal slack perturbations].
    # Round only at tie resolution to reduce spurious near-ties.
    keys = [(float(rhs[i]/direction[i]), *[float(z) for z in perturbation[i]/direction[i]], int(i)) for i in eligible]
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
    # Positive affine shifts retain best responses, while avoiding the
    # large common offset previously introduced for already-positive games.
    # Scale to bounded positive coefficients to improve basis conditioning.
    def positive_coefficients(payoff):
        lo = float(payoff.min())
        span = np.ptp(payoff.astype(np.longdouble))
        if span == 0.0:
            return np.ones_like(payoff)
        # Subtract before division to avoid unnecessary overflow for large
        # translated payoffs; the resulting values lie in [1, 2].
        return np.asarray(1.0 + (payoff.astype(np.longdouble) - np.longdouble(lo)) / span, dtype=float)
    aa = positive_coefficients(a)
    bb = positive_coefficients(b)
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
            certificate = certify_bimatrix(a, b, x, y, absolute_tolerance=tol, relative_tolerance=1e-12)
            gap = certificate['nash_gap']
            if not certificate['valid']: raise ArithmeticError(f'Numerical certificate rejected: gap={gap}, simplex={certificate["simplex_error"]:.6g}')
            if not any(np.max(abs(x-np.array(e['row_probabilities']))) <= tol and np.max(abs(y-np.array(e['col_probabilities']))) <= tol for e in results):
                results.append({'row_probabilities':x.tolist(),'col_probabilities':y.tolist(), 'expected_payoffs':expected_payoffs(a, b, x, y), 'nash_gap':gap,'certificate':certificate,'initial_label':initial,'pivots':step+1})
        except (ArithmeticError,ValueError,np.linalg.LinAlgError,OverflowError) as exc:
            failures.append({'initial_label':initial,'error':str(exc)})
    return {'method':'independent_lemke_howson','equilibria':results,'failures':failures,'labels_attempted':max_labels,'complete':False}
