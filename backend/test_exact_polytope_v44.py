"""v44 exact geometry: independently validate vertices and dimensions."""
from fractions import Fraction as F
import numpy as np
import pytest
from scipy.optimize import linprog
from exact_nash_set import solve_exact_nash_set
from exact_polytope_vertices import (
    VertexBudgetExceeded, _rank, _solve_square, enumerate_polytope_vertices)
from exact_support import solve_exact_support, ExactWorkLimit
from test_exact_nash_set_v43 import membership


def test_zero_5x5_complete_continuum_and_closure_vertices():
    z=[[0]*5 for _ in range(5)]
    with pytest.raises(ValueError):
        solve_exact_nash_set(z,z)
    r=solve_exact_nash_set(z,z,max_actions=5,max_pairs=961,include_vertices=True)
    assert r['equilibrium_set_complete'] is True
    assert r['support_pairs_checked']==961
    assert r['nonempty_support_cells']==961
    assert 0 < r['vertex_candidates_tested'] < 50000
    for family in r['equilibrium_families']:
        k=len(family['support']['row']); j=len(family['support']['col'])
        assert family['product_affine_dimension']==k+j-2
        assert family['product_closure_vertex_count']==k*j
        for field,active in [('row_strategy_polytope',k),('col_strategy_polytope',j)]:
            v=family[field]['closure_vertices_exact']
            assert len(v)==active
            assert {tuple(map(F,x)) for x in v}=={
                tuple(F(int(i==t)) for i in range(active)) for t in range(active)}


def test_nonstrict_closure_can_contain_points_outside_strict_cell():
    a=[[0,0,0],[0,0,0],[-1,-1,-1]]
    b=[[3,0,0],[3,0,0],[3,0,0]]
    r=solve_exact_nash_set(a,b,include_vertices=True)
    family=next(f for f in r['equilibrium_families'] if f['support']['row']==[0,1])
    assert family['product_affine_dimension']==1
    assert family['row_strategy_polytope']['closure_vertices_exact']==[['0','1'],['1','0']]
    assert family['row_strategy_polytope']['affine_dimension']==1
    assert membership(r['equilibrium_families'],(F(0),F(1),F(0)),(F(1),F(0),F(0)))==1
    assert membership([family],(F(0),F(1),F(0)),(F(1),F(0),F(0)))==0


def test_independent_lp_objectives_match_vertex_optima():
    rng=np.random.default_rng(4402)
    for n in (2,3,4):
        for _ in range(5):
            a=rng.integers(-3,4,size=(n,n))
            b=rng.integers(-3,4,size=(n,n))
            r=solve_exact_nash_set(a,b,include_vertices=True)
            for family in r['equilibrium_families']:
                for key in ('row_strategy_polytope','col_strategy_polytope'):
                    p=family[key]
                    k=len(p['variable_actions'])
                    ev=np.array([[float(F(x)) for x in v] for v in p['closure_vertices_exact']])
                    eq=np.array([[float(F(x)) for x in e['coefficients_exact']]
                                 for e in p['equalities_dot_coefficients_eq_zero']])
                    ub=np.array([[-float(F(x)) for x in e['coefficients_exact']]
                                 for e in p['inequalities_dot_coefficients_ge_zero']])
                    c=rng.integers(-4,5,size=k).astype(float)
                    lp=linprog(-c,A_eq=np.vstack([np.ones(k),eq]) if len(eq) else np.ones((1,k)),
                               b_eq=np.r_[1.,np.zeros(len(eq))],A_ub=ub if len(ub) else None,
                               b_ub=np.zeros(len(ub)) if len(ub) else None,
                               bounds=(0,None),method='highs')
                    assert lp.success,(p,lp.message)
                    assert abs((-lp.fun)-max(ev@c))<1e-8
                    assert 0<=p['affine_dimension']<=k-1


def test_linear_algebra_and_budget_boundary_contract():
    assert _rank([])==0
    assert _rank([[F(1),F(1)],[F(2),F(2)]])==1
    assert _rank([[F(0),F(1)],[F(1),F(0)]])==2
    assert _solve_square([[F(1),F(1)],[F(1),-F(1)]],[F(1),F(0)])==(F(1,2),F(1,2))
    assert _solve_square([[F(1),F(1)],[F(2),F(2)]],[F(1),F(2)]) is None
    with pytest.raises(VertexBudgetExceeded):
        solve_exact_nash_set([[0,0],[0,0]],[[0,0],[0,0]],include_vertices=True,max_vertex_candidates=1)
    for invalid in (0,-1,True,200001):
        with pytest.raises(ValueError):
            solve_exact_nash_set([[0]],[[0]],include_vertices=True,max_vertex_candidates=invalid)
    with pytest.raises(ValueError):
        solve_exact_nash_set([[0]],[[0]],max_actions=True)
    with pytest.raises(ValueError):
        solve_exact_nash_set([[0]],[[0]],include_vertices=1)


def test_5x5_support_optin_preserves_strict_default_and_work_budget():
    z=[[0]*5 for _ in range(5)]
    with pytest.raises(ValueError):
        solve_exact_support(z,z)
    with pytest.raises(ExactWorkLimit):
        solve_exact_support(z,z,max_actions=5,max_pairs=960)
    out=solve_exact_support(z,z,max_actions=5,max_pairs=961)
    assert out['complete_support_feasibility'] and len(out['equilibria'])==961
    with pytest.raises(ExactWorkLimit):
        solve_exact_nash_set(z,z,max_actions=5,max_pairs=961,max_operations=1)


def test_nonempty_geometry_and_vertex_fail_closed_monkeypatch(monkeypatch):
    p={'variable_actions':[0,1],
       'equalities_dot_coefficients_eq_zero':[],
       'inequalities_dot_coefficients_ge_zero':[],
       'witness_probabilities_exact':['1/2','1/2']}
    with pytest.raises(VertexBudgetExceeded):
        enumerate_polytope_vertices(p,{'tested':0,'limit':1})
    r=enumerate_polytope_vertices(p,{'tested':0,'limit':2})
    assert r['affine_dimension']==1 and len(r['closure_vertices_exact'])==2
    monkeypatch.setattr('exact_polytope_vertices._solve_square',lambda *args:None)
    with pytest.raises(AssertionError,match='no vertices'):
        enumerate_polytope_vertices(p,{'tested':0,'limit':2})


def test_inconsistent_redundant_equation_is_not_accepted_as_vertex():
    p={'variable_actions':[0,1],
       'equalities_dot_coefficients_eq_zero':[{'coefficients_exact':['1','1']}],
       'inequalities_dot_coefficients_ge_zero':[],
       'witness_probabilities_exact':['1/2','1/2']}
    # Deliberately inconsistent: simplex sum=1 conflicts with equality sum=0.
    with pytest.raises(AssertionError,match='no vertices'):
        enumerate_polytope_vertices(p,{'tested':0,'limit':10})


def test_random_five_by_five_exact_support_basis_path_against_nash_oracle():
    from test_exact_nash_set_v43 import is_nash
    rng=np.random.default_rng(44040)
    for _ in range(8):
        a=rng.integers(-5,6,size=(5,5)).tolist()
        b=rng.integers(-5,6,size=(5,5)).tolist()
        result=solve_exact_nash_set(a,b,max_actions=5,max_pairs=961,include_vertices=True)
        assert result['equilibrium_set_complete']
        assert result['support_pairs_checked']==961
        assert result['exact_work_operations']<2000000
        for family in result['equilibrium_families']:
            x=[F(0)]*5;y=[F(0)]*5
            for i,w in zip(family['support']['row'],family['row_strategy_polytope']['witness_probabilities_exact']):
                x[i]=F(w)
            for j,w in zip(family['support']['col'],family['col_strategy_polytope']['witness_probabilities_exact']):
                y[j]=F(w)
            assert is_nash(a,b,x,y)
            for poly in ('row_strategy_polytope','col_strategy_polytope'):
                obj=family[poly]
                for v in obj['closure_vertices_exact']:
                    vv=list(map(F,v))
                    assert sum(vv)==1 and min(vv)>=0
                    assert all(sum(F(c)*p for c,p in zip(item['coefficients_exact'],vv))==0
                               for item in obj['equalities_dot_coefficients_eq_zero'])
                    assert all(sum(F(c)*p for c,p in zip(item['coefficients_exact'],vv))>=0
                               for item in obj['inequalities_dot_coefficients_ge_zero'])
