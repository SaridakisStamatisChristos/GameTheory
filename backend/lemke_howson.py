"""Optional Lemke–Howson backend with explicit verification and failure reporting.

Nashpy is an optional dependency. This adapter does not claim exhaustive enumeration
or numerical correctness without independent best-response checks.
"""
import numpy as np


def solve_lemke_howson(a, b, *, tol=1e-7, max_labels=None):
    try:
        import nashpy as nash
    except ImportError as exc:
        raise RuntimeError('Install optional benchmark dependency nashpy') from exc
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or not a.size or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected nonempty equally shaped finite matrices')
    m, n = a.shape
    if max_labels is not None and (max_labels < 1 or max_labels > m+n):
        raise ValueError('Invalid max_labels')
    game = nash.Game(a, b)
    found, failures = [], []
    import warnings
    for label in range(m+n if max_labels is None else max_labels):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                x, y = game.lemke_howson(initial_dropped_label=label)
            x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            if x.shape != (m,) or y.shape != (n,) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Invalid strategy dimensions or nonfinite values')
            if min(x.min(), y.min()) < -tol or abs(x.sum()-1) > tol or abs(y.sum()-1) > tol:
                raise ValueError('Invalid probability simplex')
            gap = max(0., float(np.max(a@y)-x@a@y), float(np.max(x@b)-x@b@y))
            if gap > tol:
                raise ValueError(f'Nash gap {gap:g} exceeds tolerance')
            if not any(np.max(abs(x-np.asarray(e['row_probabilities'])))<=tol and np.max(abs(y-np.asarray(e['col_probabilities'])))<=tol for e in found):
                found.append({'row_probabilities':x.tolist(),'col_probabilities':y.tolist(),
                              'expected_payoffs':[float(x@a@y),float(x@b@y)],'nash_gap':gap,'initial_label':label})
        except (ValueError, ZeroDivisionError, IndexError, FloatingPointError, RuntimeError) as exc:
            failures.append({'initial_label':label,'error':str(exc)})
    return {'equilibria':found,'method':'lemke_howson_nashpy', 'labels_attempted':m+n if max_labels is None else max_labels,
            'failures':failures,'complete':False,
            'note':'One equilibrium per successful pivot path; not an exhaustive equilibrium enumeration.'}
