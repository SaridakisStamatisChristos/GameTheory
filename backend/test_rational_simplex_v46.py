"""v46 rational-simplex regressions: independent feasibility, completeness and budgets."""
from fractions import Fraction as F
from itertools import combinations, product
import numpy as np
import pytest
from exact_rational_lp import _RationalSimplex, feasible_strict_strategy
from exact_support import ExactWorkLimit
from general_exact_search import search_exact_nash_cells
from general_nash import solve_general_nash
from scipy.optimize import linprog
from test_exact_nash_set_v43 import is_nash


def _matrix(values):
    return [[F(z) for z in row] for row in values]


def test_simplex_phase_one_feasible_and_infeasible_exactly():
    work={'steps':0,'limit':20000}
    good=_RationalSimplex([[F(1)]],[F(2)],[F(1)],work).solve()
    assert good==[F(2)]
    bad=_RationalSimplex([[F(1)],[-F(1)]],[F(0),-F(1)],[F(1)],work).solve()
    assert bad is None


def test_exhaustive_two_valued_two_by_two_single_player_supports():
    # Independent HiGHS max-min feasibility oracle; no shared exact simplex.
    for flat in product((-1,1),repeat=4):
        arr=np.asarray(flat,dtype=float).reshape((2,2))
        a=_matrix(arr.astype(int).tolist())
        for ni in (1,2):
            for I in combinations(range(2),ni):
                for nj in (1,2):
                    for J in combinations(range(2),nj):
                        w={'steps':0,'limit':20000}
                        candidate=feasible_strict_strategy(a,I,J,w)
                        k=len(J)
                        eq=[[1.]*k+[0.]]
                        rhs=[1.]
                        for i in I[1:]:
                            eq.append([*(arr[i,list(J)]-arr[I[0],list(J)]),0.])
                            rhs.append(0.)
                        ub=[]
                        for i in range(2):
                            if i not in I:
                                ub.append([*(arr[i,list(J)]-arr[I[0],list(J)]),0.])
                        for j in range(k):
                            row=np.zeros(k+1)
                            row[j]=-1;row[-1]=1
                            ub.append(row)
                        ref=linprog([0.]*k+[-1.],A_eq=np.array(eq),b_eq=rhs,
                                   A_ub=np.array(ub),b_ub=np.zeros(len(ub)),
                                   bounds=[(0,None)]*(k+1),method='highs')
                        assert (candidate is not None)==bool(ref.success and ref.x[-1]>1e-8)
                        if candidate is not None:
                            assert sum(candidate)==1 and all(v>0 for v in candidate)
                        assert w['steps']<=20000


def test_seeded_random_3x3_and_5x5_against_vertex_oracle():
    rng=np.random.default_rng(46021)
    for n,loops in ((3,12),(5,4)):
        for _ in range(loops):
            a=rng.integers(-4,5,(n,n)).astype(float)
            b=rng.integers(-4,5,(n,n)).astype(float)
            params={'max_pairs':96 if n==5 else 49,
                    'max_operations':1000000,'max_support':2 if n==5 else 3}
            simplex=search_exact_nash_cells(a,b,**params)
            reference=search_exact_nash_cells(a,b,feasibility_backend='vertices',**params)
            def supports(result):
                return {(tuple(f['support']['row']),tuple(f['support']['col']))
                        for f in result['families']}
            assert supports(simplex)==supports(reference)
            assert simplex['support_pairs_checked']==reference['support_pairs_checked']
            assert simplex['feasibility_backend']=='simplex'
            assert reference['feasibility_backend']=='vertices'
            for eq in simplex['families']:
                x=tuple(map(F,eq['row_probabilities_exact']))
                y=tuple(map(F,eq['col_probabilities_exact']))
                assert is_nash(a.tolist(),b.tolist(),x,y)


def test_zero_game_complete_exact_four_by_four():
    z=np.zeros((4,4))
    r=search_exact_nash_cells(z,z,max_pairs=225,max_operations=500000,
                              max_support=4)
    assert r['complete_equilibrium_set']
    assert r['status']=='complete'
    assert r['support_pairs_checked']==225
    assert r['feasible_cells_found']==225
    assert r['exact_operations']>0


def test_very_small_positive_probabilities_stay_exact():
    a=_matrix([[1,0],[0,1]])
    b=_matrix([[1,0],[0,F('1e-300')]])
    row=feasible_strict_strategy(b,(0,1),(0,1),{'steps':0,'limit':100000})
    col=feasible_strict_strategy(a,(0,1),(0,1),{'steps':0,'limit':100000})
    assert row is not None and col is not None
    assert min(row)==F('1e-300')/(F(1)+F('1e-300'))
    assert all(v>0 for v in (*row,*col))


def test_fail_closed_budget_and_general_portfolio_option():
    with pytest.raises(ExactWorkLimit):
        feasible_strict_strategy(_matrix([[0,0],[0,0]]),(0,1),(0,1),
                                 {'steps':0,'limit':1})
    z=np.zeros((7,8))
    limited=search_exact_nash_cells(z,z,max_pairs=12,
                                    max_operations=150000,max_support=2)
    assert not limited['complete_equilibrium_set']
    assert limited['incomplete_reason']=='pair_budget'
    assert limited['support_pairs_checked']==12
    assert limited['feasible_cells_found']==12
    with pytest.raises(ValueError):
        search_exact_nash_cells(z,z,feasibility_backend='unknown')
    with pytest.raises(ValueError):
        solve_general_nash([[0]],[[0]],exact_backend='bad')
    for backend in ('simplex','vertices'):
        complete=solve_general_nash([[0,0],[0,0]],[[0,0],[0,0]],
                                   mode='exact',exact_backend=backend,
                                   max_exact_pairs=9,max_exact_support=2)
        assert complete['equilibrium_set_complete']
        assert complete['exact_search']['feasibility_backend']==backend


def test_bland_anti_cycling_degenerate_linear_program():
    # Classical cycling-prone example: Dantzig rule cycles but Bland converges.
    a=[[F(1,2),-F(11,2),-F(5,2),F(9)],
       [F(1,2),-F(3,2),-F(1,2),F(1)],
       [F(1),F(0),F(0),F(0)]]
    budget={'steps':0,'limit':100000}
    x=_RationalSimplex(a,[0,0,1],[10,-57,-9,-24],budget).solve()
    assert x==[F(1),F(0),F(1),F(0)]
    assert sum(c*v for c,v in zip((10,-57,-9,-24),x))==1
    assert budget['steps']<100000
