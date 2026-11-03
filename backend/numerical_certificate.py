"""Scale-aware a posteriori certificates for finite bimatrix games."""
import numpy as np
from safe_numeric import json_number


def certify_bimatrix(a, b, x, y, *, absolute_tolerance=1e-7, relative_tolerance=1e-8):
    a, b, x, y = (np.asarray(v, dtype=float) for v in (a, b, x, y))
    if a.ndim != 2 or not a.size or b.shape != a.shape or x.shape != (a.shape[0],) or y.shape != (a.shape[1],):
        raise ValueError('Incompatible payoff and strategy dimensions')
    if not all(np.isfinite(v).all() for v in (a, b, x, y)):
        raise ValueError('Nonfinite input')
    if not np.isfinite([absolute_tolerance, relative_tolerance]).all() or absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError('Invalid tolerance')
    simplex_error = max(abs(float(np.sum(x, dtype=np.longdouble))-1), abs(float(np.sum(y, dtype=np.longdouble))-1), max(0., -float(x.min()), -float(y.min())))

    def gains(payoff, row_player):
        p = payoff.astype(np.longdouble)
        # Centering removes arbitrary translations before expectation arithmetic.
        p -= p.min()
        scale = p.max()
        if scale == 0:
            return 0., 0., 0., True
        # Normalize BEFORE matrix multiplication, preventing overflow even when
        # input payoffs have opposite signs near the float64 limit.
        p /= scale
        xx, yy = x.astype(np.longdouble), y.astype(np.longdouble)
        choices = p @ yy if row_player else xx @ p
        actual = xx @ choices if row_player else choices @ yy
        relative_gain = max(np.longdouble(0), choices.max() - actual)
        # Compare in normalized coordinates to avoid overflow in tolerance.
        normalized_limit = np.longdouble(relative_tolerance) + np.longdouble(absolute_tolerance) / scale
        raw_gain = relative_gain * scale
        return json_number(raw_gain), json_number(scale), json_number(normalized_limit * scale), bool(relative_gain <= normalized_limit)

    rg, rs, rt, rv = gains(a, True)
    cg, cs, ct, cv = gains(b, False)
    return {'valid': bool(simplex_error <= absolute_tolerance and rv and cv),
            'simplex_error': simplex_error, 'row_deviation_gain': rg,
            'col_deviation_gain': cg, 'nash_gap': json_number(max(np.longdouble(rg), np.longdouble(cg))),
            'row_payoff_range': rs, 'col_payoff_range': cs,
            'row_tolerance': rt, 'col_tolerance': ct}
