"""Exact semialgebraic Nash cells for finite three-player binary-action games.

Fully mixed isolated points are found by exact quadratic elimination over
Q(sqrt(discriminant)). Lower-dimensional support cells are classified with
rational linear/bilinear inequalities. Exceptional positive-dimensional
fully-mixed varieties are declared unresolved, never silently discarded.
"""
from fractions import Fraction as F
from itertools import product
from math import isqrt

from multiplayer_exact import _game


class Quadratic:
    """A + B*sqrt(D), D>=0 rational; used only when D is not a square."""
    def __init__(self, a, b, d):
        self.a, self.b, self.d = F(a), F(b), F(d)
        if self.d < 0:
            raise ValueError('Negative quadratic radicand')

    def _coerce(self, other):
        if isinstance(other, Quadratic):
            if other.d != self.d:
                raise ValueError('Incompatible quadratic fields')
            return other
        return Quadratic(other, 0, self.d)

    def __add__(self, other):
        z = self._coerce(other)
        return Quadratic(self.a + z.a, self.b + z.b, self.d)

    __radd__ = __add__

    def __neg__(self):
        return Quadratic(-self.a, -self.b, self.d)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        z = self._coerce(other)
        return Quadratic(self.a*z.a + self.b*z.b*self.d,
                         self.a*z.b + self.b*z.a, self.d)

    __rmul__ = __mul__

    def __truediv__(self, other):
        z = self._coerce(other)
        norm = z.a*z.a - z.b*z.b*self.d
        if norm == 0:
            raise ZeroDivisionError('Division by zero quadratic element')
        return Quadratic((self.a*z.a-self.b*z.b*self.d)/norm,
                         (self.b*z.a-self.a*z.b)/norm, self.d)

    def __rtruediv__(self, other):
        return self._coerce(other)/self

    def sign(self):
        a,b = self.a,self.b
        if not b:
            return (a>0)-(a<0)
        if not a:
            return (b>0)-(b<0)
        if (a>0)==(b>0):
            return (a>0)-(a<0)
        diff = a*a-b*b*self.d
        if not diff:
            return 0
        dominant = a if diff>0 else b
        return (dominant>0)-(dominant<0)


def _sign(v):
    if isinstance(v, Quadratic):
        return v.sign()
    return (v>0)-(v<0)


def _encode(v):
    if isinstance(v, Quadratic):
        if v.b:
            return {'type':'quadratic','a':str(v.a),'b':str(v.b),
                    'radicand':str(v.d)}
        v = v.a
    return {'type':'rational','value':str(v)}


def _sqrt_rational(x):
    n, d = isqrt(x.numerator), isqrt(x.denominator)
    if n*n == x.numerator and d*d == x.denominator:
        return F(n, d)
    return None


def _quadratic_roots(p):
    a,b,c = p[0],p[1],p[2]
    if c == 0:
        return [] if b == 0 else [-a/b]
    delta = b*b - 4*c*a
    if delta < 0:
        return []
    square = _sqrt_rational(delta)
    if square is not None:
        return sorted(set(((-b+square)/(2*c),(-b-square)/(2*c))))
    return [Quadratic(-b/(2*c),F(1,1)/(2*c),delta),
            Quadratic(-b/(2*c),-F(1,1)/(2*c),delta)]


def _polynomial_product(u,v):
    out=[F(0)]*3
    for i,a in enumerate(u):
        for j,b in enumerate(v):
            out[i+j]+=a*b
    return out


def _difference_coefficients(utilities, profiles):
    index={profile:i for i,profile in enumerate(profiles)}
    result=[]
    for player in range(3):
        others=[q for q in range(3) if q!=player]
        values=[]
        for a,b in ((0,0),(1,0),(0,1),(1,1)):
            profile=[0]*3
            profile[others[0]],profile[others[1]]=a,b
            profile[player]=0
            lo=utilities[player][index[tuple(profile)]]
            profile[player]=1
            hi=utilities[player][index[tuple(profile)]]
            values.append(hi-lo)
        v00,v10,v01,v11=values
        result.append((v00,v10-v00,v01-v00,v11-v10-v01+v00))
    return result


def _difference(coefficient, player, values):
    others=[q for q in range(3) if q!=player]
    y,z=values[others[0]],values[others[1]]
    a,b,c,d=coefficient
    return a+b*y+c*z+d*y*z


def _verify_point(values, utilities, profiles):
    """Direct original-payoff pure-deviation audit, independent of eliminants."""
    payoffs=[]
    for p in range(3):
        action_values=[F(0),F(0)]
        for k,profile in enumerate(profiles):
            weight=F(1)
            for other in range(3):
                if other!=p:
                    prob=values[other]
                    weight *= prob if profile[other] else 1-prob
            action_values[profile[p]] += utilities[p][k]*weight
        x=values[p]
        if (_sign(x)<0 or _sign(1-x)<0 or
                (_sign(x)>0 and _sign(1-x)>0 and _sign(action_values[1]-action_values[0])!=0) or
                (_sign(x)==0 and _sign(action_values[1]-action_values[0])>0) or
                (_sign(1-x)==0 and _sign(action_values[1]-action_values[0])<0)):
            return None
        payoffs.append((1-x)*action_values[0]+x*action_values[1])
    return payoffs


def _point(values, u, profiles):
    expected=_verify_point(values,u,profiles)
    if expected is None:
        return None
    return {
        'action1_probabilities_exact': [_encode(v) for v in values],
        'strategies_exact': [[_encode(1-v),_encode(v)] for v in values],
        'expected_payoffs_exact': [_encode(v) for v in expected],
        'support': [[0] if _sign(v)==0 else [1] if _sign(1-v)==0 else [0,1]
                    for v in values],
        'certificate':'direct exact algebraic original-payoff unilateral-deviation audit',
    }


def _one_mixed(pattern, coefficients, u, profiles):
    p=pattern.index('m')
    values=[F(int(c)) if c!='m' else F(0) for c in pattern]
    if _difference(coefficients[p],p,values)!=0:
        return 'empty',None
    lo,hi=F(0),F(1)
    for q in range(3):
        if q==p:
            continue
        values[p]=F(0)
        c0=_difference(coefficients[q],q,values)
        values[p]=F(1)
        c1=_difference(coefficients[q],q,values)-c0
        if pattern[q]=='0':
            c0,c1=-c0,-c1
        if c1==0:
            if c0<0:
                return 'empty',None
        elif c1>0:
            lo=max(lo,-c0/c1)
        else:
            hi=min(hi,-c0/c1)
    if lo>hi or hi<=0 or lo>=1:
        return 'empty',None
    if lo==hi:
        values[p]=lo
        point=_point(values,u,profiles)
        if point is None:
            raise AssertionError('Failed exact one-mixed interval endpoint audit')
        return 'point',point
    return 'family',{
        'kind':'one_parameter_interval',
        'support': [[0,1] if c=='m' else [int(c)] for c in pattern],
        'free_player':p,
        'action1_probability_lower_exact':str(lo),
        'action1_probability_upper_exact':str(hi),
        'lower_inclusive':lo>0,
        'upper_inclusive':hi<1,
        'other_players_action1_probabilities_exact':{
            str(q):str(F(int(pattern[q]))) for q in range(3) if q!=p},
        'certificate':'exact rational indifference and two affine best-response inequalities',
    }


def _two_mixed(pattern, coefficients, u, profiles):
    mixed=[p for p in range(3) if pattern[p]=='m']
    pure=next(p for p in range(3) if p not in mixed)
    values=[F(int(c)) if c!='m' else F(0) for c in pattern]
    fixed={}
    for p in mixed:
        other=next(q for q in mixed if q!=p)
        values[other]=F(0)
        c0=_difference(coefficients[p],p,values)
        values[other]=F(1)
        c1=_difference(coefficients[p],p,values)-c0
        if c1==0:
            if c0!=0:
                return 'empty',None
        else:
            root=-c0/c1
            if not 0<root<1:
                return 'empty',None
            fixed[other]=root
    free=[p for p in mixed if p not in fixed]

    def signed_pure(point):
        v=[fixed.get(p, F(int(pattern[p])) if p==pure else F(0))
           for p in range(3)]
        for p,x in zip(free,point):
            v[p]=x
        delta=_difference(coefficients[pure],pure,v)
        return delta if pattern[pure]=='1' else -delta

    corners={tuple(t):signed_pure(t) for t in product((F(0),F(1)),repeat=len(free))}
    maximum=max(corners.values())
    if maximum<0 or (maximum==0 and any(v<0 for v in corners.values())):
        return 'empty',None
    if not free:
        point=_point([fixed[p] if p in fixed else F(int(pattern[p]))
                      for p in range(3)],u,profiles)
        if point is None:
            raise AssertionError('Failed exact two-mixed point audit')
        return 'point',point
    zero=(F(0),)*len(free)
    constant=corners[zero]
    poly={'constant':str(constant)}
    for i,p in enumerate(free):
        pt=list(zero);pt[i]=F(1)
        poly[f'p{p}']=str(corners[tuple(pt)]-constant)
    if len(free)==2:
        poly[f'p{free[0]}_p{free[1]}']=str(
            corners[(F(1),F(1))]-corners[(F(1),F(0))]
            -corners[(F(0),F(1))]+constant)
    return 'family',{
        'kind':'bilinear_semialgebraic_cell',
        'support':[[0,1] if c=='m' else [int(c)] for c in pattern],
        'free_action1_probability_players':free,
        'strict_bounds':'0 < p[player] < 1 for each free player',
        'fixed_action1_probabilities_exact':{
            str(p):str(v) for p,v in fixed.items()},
        'pure_player':pure,'pure_action':int(pattern[pure]),
        'pure_best_response_polynomial_ge_zero_exact':poly,
        'certificate':'exact indifference equations and convex-combination corner feasibility test',
    }


def _three_mixed(coefficients, u, profiles):
    a,b,c=coefficients
    ny=(-a[0],-a[2]);dy=(a[1],a[3])
    nx=(-b[0],-b[2]);dx=(b[1],b[3])
    terms=[(c[0],dx,dy),(c[1],nx,dy),
           (c[2],ny,dx),(c[3],nx,ny)]
    polynomial=[F(0)]*3
    for weight,left,right in terms:
        part=_polynomial_product(left,right)
        for i in range(3):
            polynomial[i]+=weight*part[i]
    if not any(polynomial):
        return [],'identically_zero_eliminant'
    if not any(dx) or not any(dy):
        return [],'nonisolated_elimination_denominator'
    candidates=[]
    unknown=False
    for z in _quadratic_roots(polynomial):
        if _sign(z)<=0 or _sign(1-z)<=0:
            continue
        dxx=dx[0]+dx[1]*z; dyy=dy[0]+dy[1]*z
        nxx=nx[0]+nx[1]*z; nyy=ny[0]+ny[1]*z
        if _sign(dxx)==0 or _sign(dyy)==0:
            if (_sign(dxx)==0 and _sign(nxx)==0) or (
                    _sign(dyy)==0 and _sign(nyy)==0):
                unknown=True
            continue
        x=nxx/dxx
        y=nyy/dyy
        if not all(_sign(p)>0 and _sign(1-p)>0 for p in (x,y)):
            continue
        point=_point([x,y,z],u,profiles)
        if point is None:
            raise AssertionError('Quadratic elimination produced non-Nash point')
        candidates.append(point)
    return candidates,'singular_denominator_root' if unknown else None


def solve_certified_binary_multiplayer(payoffs, *, max_cells=27):
    """Exact support-cell decomposition, complete only if no unresolved cells."""
    if type(max_cells) is not int or not 1<=max_cells<=27:
        raise ValueError('max_cells must be an integer from 1 to 27')
    shape,profiles,utilities=_game(payoffs)
    if shape!=(2,2,2):
        raise ValueError('v60 certified polynomial engine requires exactly three players with two actions each')
    coeffs=_difference_coefficients(utilities,profiles)
    patterns=sorted(product(('0','1','m'),repeat=3),
                    key=lambda p:(p.count('m'),p))
    points=[];families=[];unresolved=[];empty=0
    for pattern in patterns[:max_cells]:
        count=pattern.count('m')
        if count==0:
            values=[F(int(c)) for c in pattern]
            result=_point(values,utilities,profiles)
            kind,entry=('point',result) if result is not None else ('empty',None)
        elif count==1:
            kind,entry=_one_mixed(pattern,coeffs,utilities,profiles)
        elif count==2:
            kind,entry=_two_mixed(pattern,coeffs,utilities,profiles)
        else:
            solutions,reason=_three_mixed(coeffs,utilities,profiles)
            points.extend(solutions)
            kind,entry=('unresolved',reason) if reason else ('solved',None)
        if kind=='point':
            points.append(entry)
        elif kind=='family':
            families.append(entry)
        elif kind=='empty':
            empty+=1
        elif kind=='unresolved':
            unresolved.append({'support_pattern':list(pattern),
                               'reason':entry,
                               'exact_indifference_coefficients':[
                                   [str(v) for v in row] for row in coeffs]})
    complete=max_cells==27 and not unresolved
    return {
        'status':'complete_exact_semialgebraic_nash_set' if complete
                 else 'partial_exact_support_decomposition',
        'complete_nash_set':complete,
        'players':3,'action_counts':[2,2,2],
        'support_cells_total':27,'support_cells_processed':max_cells,
        'empty_support_cells':empty,
        'isolated_equilibria':points,
        'continuous_equilibrium_cells':families,
        'unresolved_support_cells':unresolved,
        'unvisited_support_cells':27-max_cells,
        'input_model':'decimal strings of validated float64 payoffs',
        'certificate':'exact rational and quadratic-field incentive checks; conditional completeness by exhaustion of every positive-support cell',
        'limitation':'Singular or identically-zero fully-mixed eliminants are not enumerated; unresolved support cells prevent full completeness.',
    }
