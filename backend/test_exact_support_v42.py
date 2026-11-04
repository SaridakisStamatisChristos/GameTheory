"""Independent mathematical oracle checks for exact support feasibility."""
import itertools
from fractions import Fraction
import numpy as np
import pytest
from scipy.optimize import linprog
from exact_support import ExactWorkLimit, solve_exact_support

def reference_support_feasible(a,b,rows,cols):
    """Independent max-min SciPy LP reference for small integer games."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    def positive_mixture(p,best,support):
        k=len(support)
        eq=[[ *([1.]*k), 0. ]]
        rhs=[1.]
        first=best[0]
        for i in best[1:]:
            eq.append([*(p[i,support]-p[first,support]),0.]);rhs.append(0.)
        le=[]
        for i in range(len(p)):
            if i not in best:
                le.append([*(p[i,support]-p[first,support]),0.])
        for j in range(k):
            v=np.zeros(k+1);v[j]=-1;v[-1]=1;le.append(v)
        r=linprog([*([0.]*k),-1.],A_eq=np.asarray(eq),b_eq=np.asarray(rhs),
                  A_ub=np.asarray(le),b_ub=np.zeros(len(le)),
                  bounds=[(None,None)]*k+[(0,None)],method='highs')
        return bool(r.success and r.x[-1]>1e-7)
    return positive_mixture(a,rows,cols) and positive_mixture(b.T,cols,rows)

def _assert_exact_certificate(a,b,result):
    af=[[Fraction(str(float(v))) for v in row] for row in a]
    bf=[[Fraction(str(float(v))) for v in row] for row in b]
    m,n=len(af),len(af[0])
    assert result['complete_support_feasibility'] is True
    assert result['all_equilibria_enumerated'] is False
    assert result['support_pairs_checked']==(2**m-1)*(2**n-1)
    supports=set()
    for e in result['equilibria']:
        x=[Fraction(s) for s in e['row_probabilities_exact']]
        y=[Fraction(s) for s in e['col_probabilities_exact']]
        assert sum(x)==sum(y)==1
        assert min(x)>=0 and min(y)>=0
        I=tuple(i for i,t in enumerate(x) if t>0)
        J=tuple(j for j,t in enumerate(y) if t>0)
        assert I==tuple(e['support']['row'])
        assert J==tuple(e['support']['col'])
        assert (I,J) not in supports
        supports.add((I,J))
        row=[sum(af[i][j]*y[j] for j in range(n)) for i in range(m)]
        col=[sum(x[i]*bf[i][j] for i in range(m)) for j in range(n)]
        assert all(row[i]==max(row) for i in I)
        assert all(col[j]==max(col) for j in J)
    assert result['feasible_support_pairs']==len(supports)
    return supports

def test_known_and_all_degenerate_supports():
    a=[[1,-1],[-1,1]];b=[[-1,1],[1,-1]]
    out=solve_exact_support(a,b)
    assert _assert_exact_certificate(a,b,out)=={((0,1),(0,1))}
    for n in (1,2,3,4):
        m=[[0]*n for _ in range(n)]
        r=solve_exact_support(m,m)
        _assert_exact_certificate(m,m,r)
        assert len(r['equilibria'])==(2**n-1)**2

def test_rare_positive_support():
    a=[[1,0],[0,1]];b=[[1,0],[0,10**12]]
    r=solve_exact_support(a,b)
    got=_assert_exact_certificate(a,b,r)
    assert ((0,1),(0,1)) in got
    assert any('1000000000001' in str(e['row_probabilities_exact']) for e in r['equilibria'])

def test_exhaustive_two_valued_2x2_against_external_lp():
    for values in itertools.product((-1,1),repeat=8):
        a=[values[:2],values[2:4]]
        b=[values[4:6],values[6:]]
        r=solve_exact_support(a,b)
        got=_assert_exact_certificate(a,b,r)
        expected=set()
        for ki in (1,2):
            for I in itertools.combinations(range(2),ki):
                for kj in (1,2):
                    for J in itertools.combinations(range(2),kj):
                        if reference_support_feasible(a,b,I,J):
                            expected.add((I,J))
        assert got==expected

def test_seeded_3x3_and_4x4_supports_against_independent_lp():
    rng=np.random.default_rng(41021)
    for n in (3,4):
        for _ in range(24 if n==3 else 5):
            a=rng.integers(-4,5,size=(n,n))
            b=rng.integers(-4,5,size=(n,n))
            got=_assert_exact_certificate(a,b,solve_exact_support(a,b))
            expected=set()
            for ki in range(1,n+1):
                for I in itertools.combinations(range(n),ki):
                    for kj in range(1,n+1):
                        for J in itertools.combinations(range(n),kj):
                            if reference_support_feasible(a,b,I,J):
                                expected.add((I,J))
            assert got==expected,(n,a.tolist(),b.tolist(),got^expected)

def test_unequal_support_and_budget_rejection():
    a=[[2,2,2],[0,0,0],[0,0,0]]
    b=[[1,1,0],[0,0,0],[0,0,0]]
    supports=_assert_exact_certificate(a,b,solve_exact_support(a,b))
    assert ((0,),(0,1)) in supports
    with pytest.raises(ExactWorkLimit):
        solve_exact_support([[0]*3]*3,[[0]*3]*3,max_pairs=8)
    with pytest.raises(ExactWorkLimit):
        solve_exact_support([[0]*3]*3,[[0]*3]*3,max_operations=1)
    with pytest.raises(ValueError):
        solve_exact_support([[0]*5]*5,[[0]*5]*5)
    with pytest.raises(ValueError):
        solve_exact_support([[0,np.inf]],[[0,0]])

def test_float_projection_underflow_is_reported():
    a=[[1,0],[0,1]]
    b=[[1e300,0],[0,1e-300]]
    r=solve_exact_support(a,b)
    _assert_exact_certificate(a,b,r)
    assert any(not e['float_projection_support_preserving'] for e in r['equilibria'])
