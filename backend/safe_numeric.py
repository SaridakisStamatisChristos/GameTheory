"""Finite JSON-safe payoff arithmetic, including values beyond float64 range."""
import numpy as np


def json_number(value):
    v = np.longdouble(value)
    if not np.isfinite(v):
        raise ValueError('Nonfinite numeric diagnostic')
    if abs(v) <= np.finfo(float).max:
        return float(v)
    return np.format_float_scientific(v, precision=18, unique=False, trim='k')


def expected_payoffs(a, b, x, y):
    xx = np.asarray(x, dtype=np.longdouble)
    yy = np.asarray(y, dtype=np.longdouble)
    result = []
    for payoff in (a, b):
        p = np.asarray(payoff, dtype=np.longdouble)
        # Long-double arithmetic avoids overflow of float64 intermediate products.
        result.append(json_number(xx @ p @ yy))
    return result
