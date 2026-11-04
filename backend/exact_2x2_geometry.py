"""Complete exact-rational Nash-equilibrium geometry for 2x2 decimal games.

The set is a finite union of closed points, segments, and rectangles in
(x, y) = (Pr(row 0), Pr(column 0)). Rationals represent the decimal strings
of the already-validated float64 payoffs used by the public API.
"""
from fractions import Fraction

ZERO, ONE = Fraction(0), Fraction(1)


def _where_linear(v0, v1, sign):
    """Closed interval where affine f(0)=v0 and f(1)=v1 satisfies sign 0."""
    if v0 == v1:
        valid = (v0 == 0 if sign == "eq" else
                 v0 <= 0 if sign == "le" else v0 >= 0)
        return (ZERO, ONE) if valid else None
    root = -v0 / (v1 - v0)
    if sign == "eq":
        return (root, root) if ZERO <= root <= ONE else None
    increasing = v1 > v0
    lo, hi = ((ZERO, root) if increasing else (root, ONE)) if sign == "le" else (
        (root, ONE) if increasing else (ZERO, root))
    lo, hi = max(lo, ZERO), min(hi, ONE)
    return (lo, hi) if lo <= hi else None


def _intersect(left, right):
    if right is None:
        return None
    lo, hi = max(left[0], right[0]), min(left[1], right[1])
    return (lo, hi) if lo <= hi else None


def _contains(outer, inner):
    return all(a[0] <= b[0] and a[1] >= b[1] for a, b in zip(outer, inner))


def complete_nash_geometry(aa, bb):
    """Return the full finite union of equilibria for rational 2x2 payoffs.

    With x=Pr(row 0) and y=Pr(col 0), let D_A(y) be row 0's
    expected-payoff advantage, D_B(x) column 0's advantage.
    Row best response: x=0 -> D_A<=0; x=1 -> D_A>=0;
    0<x<1 -> D_A=0. Column conditions are symmetric.
    These nine product-region cases have affine interval constraints.
    The closure of each nonempty interior case is also Nash by
    continuity, so closed rectangles represent the entire set.
    """
    da0 = aa[0][0] - aa[1][0]
    da1 = aa[0][1] - aa[1][1]
    db0 = bb[0][0] - bb[0][1]
    db1 = bb[1][0] - bb[1][1]
    modes = (("zero", (ZERO, ZERO), "le"),
             ("one", (ONE, ONE), "ge"),
             ("interior", (ZERO, ONE), "eq"))
    rectangles = []
    for xm, xdomain, row_relation in modes:
        ycondition = _where_linear(da1, da0, row_relation)
        if ycondition is None:
            continue
        for ym, ydomain, col_relation in modes:
            xcondition = _where_linear(db1, db0, col_relation)
            xspan = _intersect(xdomain, xcondition)
            yspan = _intersect(ydomain, ycondition)
            if xspan is None or yspan is None:
                continue
            # Boundary-only intersections are covered by pure modes.
            if xm == "interior" and not (xspan[1] > ZERO and xspan[0] < ONE):
                continue
            if ym == "interior" and not (yspan[1] > ZERO and yspan[0] < ONE):
                continue
            # Relative interiors of distinct support modes are disjoint;
            # their nonempty closures cannot be identical.
            rectangles.append((xspan, yspan))
    maximal = [r for r in rectangles if not any(
        r != other and _contains(other, r) for other in rectangles)]
    maximal.sort(key=lambda r: tuple(v for interval in r for v in interval))
    return [
        {
            "row_probability_bounds": [float(xspan[0]), float(xspan[1])],
            "col_probability_bounds": [float(yspan[0]), float(yspan[1])],
            "row_probability_bounds_exact": [str(xspan[0]), str(xspan[1])],
            "col_probability_bounds_exact": [str(yspan[0]), str(yspan[1])],
            "dimension": int(xspan[0] < xspan[1]) + int(yspan[0] < yspan[1]),
        }
        for xspan, yspan in maximal
    ]
