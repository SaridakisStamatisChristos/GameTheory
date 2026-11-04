"""Exact rational support feasibility for finite two-player bimatrix games.

For each feasible strict support pair (I,J), returns one exact Nash witness.
Does not enumerate all equilibria when a support admits a continuum.
Rational inputs are decimal strings of validated binary64 payoffs.
"""
from fractions import Fraction
from itertools import combinations
from math import gcd, lcm
import numpy as np

class ExactWorkLimit(ValueError):
    """Declared exact arithmetic work budget exceeded (no partial certificate)."""

def _normalize(coeffs, rhs, strict):
    values = (*coeffs, rhs)
    den = lcm(*(v.denominator for v in values))
    integers = [v.numerator * (den // v.denominator) for v in values]
    common = gcd(*integers)
    if common:
        integers = [v // common for v in integers]
    return tuple(Fraction(x) for x in integers[:-1]), Fraction(integers[-1]), strict

def _deduplicate(rows):
    seen = {}
    for coeffs, rhs, strict in rows:
        coeffs, rhs, strict = _normalize(coeffs, rhs, strict)
        if not any(coeffs):
            if rhs > 0 or (rhs == 0 and strict):
                return None
            continue
        key = (coeffs, rhs)
        seen[key] = seen.get(key, False) or strict
    return [(coeffs, rhs, flag) for (coeffs, rhs), flag in seen.items()]

def _choose_value(rows, known):
    lower = upper = None
    lower_strict = upper_strict = False
    for coeffs, rhs, strict in rows:
        c = coeffs[0]
        if c == 0:
            total = sum(v * x for v, x in zip(coeffs[1:], known))
            if not (total > rhs if strict else total >= rhs):
                raise AssertionError('Back substitution violates a zero-coefficient row')
            continue
        remaining = sum(v * x for v, x in zip(coeffs[1:], known))
        bound = (rhs - remaining) / c
        if c > 0:
            if lower is None or bound > lower:
                lower, lower_strict = bound, strict
            elif bound == lower:
                lower_strict |= strict
        else:
            if upper is None or bound < upper:
                upper, upper_strict = bound, strict
            elif bound == upper:
                upper_strict |= strict
    if lower is not None and upper is not None:
        if lower > upper or (lower == upper and (lower_strict or upper_strict)):
            raise AssertionError('Fourier-Motzkin back-substitution became infeasible')
        return (lower + upper) / 2 if lower < upper else lower
    if lower is not None:
        return lower + 1 if lower_strict else lower
    if upper is not None:
        return upper - 1 if upper_strict else upper
    return Fraction(0)

def _strict_rational_feasible(inequalities, work):
    """Exact constructive Fourier-Motzkin elimination for >= and >."""
    rows = _deduplicate(inequalities)
    if rows is None:
        return None
    history = []
    while rows and len(rows[0][0]):
        work['steps'] += len(rows)
        if work['steps'] > work['limit']:
            raise ExactWorkLimit('Exact feasibility work budget exceeded')
        history.append(rows)
        pos = [row for row in rows if row[0][0] > 0]
        neg = [row for row in rows if row[0][0] < 0]
        zero = [(coeffs[1:], rhs, strict) for coeffs, rhs, strict in rows if coeffs[0] == 0]
        work['steps'] += len(pos) * len(neg)
        if work['steps'] > work['limit']:
            raise ExactWorkLimit('Exact feasibility work budget exceeded')
        generated = list(zero)
        for pc, pr, ps in pos:
            for nc, nr, ns in neg:
                pscale, nscale = -nc[0], pc[0]
                generated.append((
                    tuple(pscale*p + nscale*n for p,n in zip(pc[1:],nc[1:])),
                    pscale*pr + nscale*nr, ps or ns))
        rows = _deduplicate(generated)
        if rows is None:
            return None
    if not history:
        return []
    solution = []
    for old_rows in reversed(history):
        solution.insert(0, _choose_value(old_rows, solution))
    if not all(
        sum(a*x for a,x in zip(c,solution)) > rhs if strict else
        sum(a*x for a,x in zip(c,solution)) >= rhs
        for c,rhs,strict in inequalities
    ):
        raise AssertionError('Constructed rational witness violates an inequality')
    return solution

def _feasible_strategy(payoff, best_responses, support, work):
    """Find positive probability on support with prescribed best responses."""
    k = len(support)
    size = len(payoff)
    constraints = []
    one = Fraction(1)
    zero = Fraction(0)
    for j in range(k):
        row = tuple(one if i==j else zero for i in range(k))
        constraints.append((row, zero, True))
    constraints.append((tuple([one]*k), one, False))
    constraints.append((tuple([-one]*k), -one, False))
    base = best_responses[0]
    for i in range(size):
        if i == base:
            continue
        coeffs = tuple(payoff[base][j] - payoff[i][j] for j in support)
        constraints.append((coeffs, zero, False))
        if i in best_responses:
            constraints.append((tuple(-v for v in coeffs), zero, False))
    witness = _strict_rational_feasible(constraints, work)
    if witness is None:
        return None
    if sum(witness) != one or not all(z>zero for z in witness):
        raise AssertionError('Unexpected invalid exact probability witness')
    return witness

def _verify_exact_equilibrium(a,b,x,y):
    """Direct independent rational Nash-deviation certificate."""
    if sum(x) != 1 or sum(y) != 1 or not all(t>=0 for t in (*x,*y)):
        return False
    row_util = [sum(a[i][j]*y[j] for j in range(len(y))) for i in range(len(x))]
    col_util = [sum(x[i]*b[i][j] for i in range(len(x))) for j in range(len(y))]
    return (sum(x[i]*row_util[i] for i in range(len(x))) == max(row_util)
            and sum(y[j]*col_util[j] for j in range(len(y))) == max(col_util))

def solve_exact_support(a, b, *, max_pairs=225, max_operations=100000, max_actions=4):
    """Exact feasible-support classification for at most 4x4 by default, opt-in 5x5.

    Enumerates (2**m-1)(2**n-1) support pairs. Raises on exhausted budgets
    rather than returning partial feasibility as complete.
    """
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.ndim != 2 or a.size == 0 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected matching nonempty finite payoff matrices')
    m,n=a.shape
    if type(max_actions) is not int or max_actions not in (4,5):
        raise ValueError('max_actions must be 4 or 5')
    if max(m,n)>max_actions or min(m,n)<1:
        raise ValueError(f'Exact support feasibility limited to {max_actions}x{max_actions} games')
    if not (type(max_pairs) is int and 1<=max_pairs<=(2**max_actions-1)**2 and type(max_operations) is int
            and 1<=max_operations<=(500000 if max_actions==4 else 2000000)):
        raise ValueError('Invalid exact support work budget')
    pairs=(2**m-1)*(2**n-1)
    if pairs>max_pairs:
        raise ExactWorkLimit(f'{pairs} support pairs exceed max_pairs={max_pairs}')
    af=[[Fraction(str(value)) for value in row] for row in a]
    bf=[[Fraction(str(value)) for value in row] for row in b]
    bt=list(map(list,zip(*bf)))
    found=[]
    work={'steps':0,'limit':max_operations}
    for ki in range(1,m+1):
        for I in combinations(range(m),ki):
            for kj in range(1,n+1):
                for J in combinations(range(n),kj):
                    y=_feasible_strategy(af,I,J,work)
                    if y is None:
                        continue
                    x=_feasible_strategy(bt,J,I,work)
                    if x is None:
                        continue
                    fullx=[Fraction(0)]*m
                    fully=[Fraction(0)]*n
                    for i,z in zip(I,x):fullx[i]=z
                    for j,z in zip(J,y):fully[j]=z
                    if not _verify_exact_equilibrium(af,bf,fullx,fully):
                        raise AssertionError('An exact support witness failed Nash certification')
                    xp=[float(z) for z in fullx]
                    yp=[float(z) for z in fully]
                    found.append({
                        'row_probabilities':xp,
                        'col_probabilities':yp,
                        'row_probabilities_exact':[str(z) for z in fullx],
                        'col_probabilities_exact':[str(z) for z in fully],
                        'float_projection_support_preserving':bool(all(p>0 for p,z in zip(xp,fullx) if z>0) and
                                                                 all(p>0 for p,z in zip(yp,fully) if z>0)),
                        'support':{'row':list(I),'col':list(J)},
                        'nash_gap_exact':'0',
                        'certificate':'exact rational best-response inequalities',
                    })
    if not found:
        raise AssertionError('Finite game unexpectedly has no exact Nash witness')
    return {
        'equilibria':found,
        'feasible_support_pairs':len(found),
        'support_pairs_checked':pairs,
        'complete_support_feasibility':True,
        'all_equilibria_enumerated':False,
        'exact_work_operations':work['steps'],
        'exact_input_model':'decimal strings of validated binary64 payoffs',
        'note':'One exact Nash witness per feasible support pair; continuous equilibrium families are not enumerated. '
               'Exact rational strings are authoritative when float probability projection loses precision.',
    }
