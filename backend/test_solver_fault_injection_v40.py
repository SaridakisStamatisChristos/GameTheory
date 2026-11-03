"""Pivot, LP, numerical, and input failures must not be advertised as equilibria."""
from types import SimpleNamespace
import builtins
import numpy as np
import pytest
from fastapi import HTTPException
import main,lemke_howson,independent_lh,research,advanced
from analytic_2x2 import solve_2x2
from approximate import approximate_nash
from hybrid_nash import solve_hybrid
from support_lp import solve_support_lp
from safe_numeric import json_number
A=[[1,-1],[-1,1]]
B=[[-1,1],[1,-1]]


def test_optional_nashpy_import_failure(monkeypatch):
    orig=builtins.__import__
    def intercept(name,*a,**kw):
        if name=='nashpy':raise ImportError('missing')
        return orig(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',intercept)
    with pytest.raises(RuntimeError,match='optional benchmark dependency nashpy'):
        lemke_howson.solve_lemke_howson(A,B)


@pytest.mark.parametrize('a,b,max_labels',[([[1],[2]],[[1]],None),([[float('inf')]],[[0]],None),(A,B,0),(A,B,5)])
def test_nashpy_adapter_input_validation(a,b,max_labels):
    with pytest.raises(ValueError):lemke_howson.solve_lemke_howson(a,b,max_labels=max_labels)


def test_nashpy_adapter_rejects_bad_profile_and_retains_valid_one(monkeypatch):
    import nashpy
    class FakeGame:
        def __init__(self,a,b):pass
        def lemke_howson(self,*,initial_dropped_label):
            return [([.5],[.5,.5]),([.8,.8],[.5,.5]),([1.,0.],[1.,0.]),([.5,.5],[.5,.5])][initial_dropped_label]
    monkeypatch.setattr(nashpy,'Game',FakeGame)
    result=lemke_howson.solve_lemke_howson(A,B)
    assert len(result['failures'])==3 and len(result['equilibria'])==1
    assert result['equilibria'][0]['nash_gap']==0


def test_nashpy_adapter_isolates_exception_and_deduplicates(monkeypatch):
    import nashpy
    class FakeGame:
        def __init__(self,a,b):pass
        def lemke_howson(self,*,initial_dropped_label):
            if initial_dropped_label==1:raise ZeroDivisionError('pivot singular')
            return [.5,.5],[.5,.5]
    monkeypatch.setattr(nashpy,'Game',FakeGame)
    result=lemke_howson.solve_lemke_howson(A,B)
    assert len(result['failures'])==1 and len(result['equilibria'])==1
    assert 'pivot singular' in result['failures'][0]['error']


def test_independent_pivot_unbounded_and_bad_budgets():
    with pytest.raises(ArithmeticError,match='Unbounded'):
        independent_lh._pivot(np.array([[1.,0.]]),[0],[0],1)
    for kw in [dict(max_labels=0),dict(max_labels=10),dict(max_labels='two'),dict(max_pivots=0)]:
        with pytest.raises(ValueError):independent_lh.solve_independent_lh(A,B,**kw)


def test_independent_solver_pivot_budget_and_unbounded_failure(monkeypatch):
    result=independent_lh.solve_independent_lh(A,B,max_pivots=1)
    assert any('Pivot budget exhausted' in x['error'] for x in result['failures'])
    def fail(*a,**kw):raise ArithmeticError('Unbounded complementary pivot ray')
    monkeypatch.setattr(independent_lh,'_pivot',fail)
    result=independent_lh.solve_independent_lh(A,B,max_labels=2)
    assert len(result['failures'])==2 and not result['equilibria']


def test_independent_solver_certificate_failure_produces_no_false_solution(monkeypatch):
    monkeypatch.setattr(independent_lh,'certify_bimatrix',lambda *a,**kw:{'valid':False,'simplex_error':.1,'nash_gap':1.})
    result=independent_lh.solve_independent_lh(A,B)
    assert not result['equilibria']
    assert any('Numerical certificate rejected' in x['error'] for x in result['failures'])


def test_zero_sum_lp_failure_is_controlled(monkeypatch):
    monkeypatch.setattr(main,'linprog',lambda *a,**kw:SimpleNamespace(success=False))
    with pytest.raises(HTTPException,match='inconclusive'):
        main.zero_sum(main.ZeroSum(payoffs=A))


def test_zero_sum_independent_certificate_rejects_false_lp_success(monkeypatch):
    calls=[]
    def bogus(*a,**kw):
        calls.append(True)
        return SimpleNamespace(success=True,x=np.array([1.,0.,0.]) if len(calls)==1 else np.array([0.,1.,0.]))
    monkeypatch.setattr(main,'linprog',bogus)
    with pytest.raises(HTTPException,match='certificate failed'):
        main.zero_sum(main.ZeroSum(payoffs=A))


def test_correlated_lp_failures_and_original_payoff_certificate(monkeypatch):
    monkeypatch.setattr(research,'linprog',lambda *a,**kw:SimpleNamespace(success=False))
    with pytest.raises(HTTPException,match='inconclusive'):
        research.correlated_equilibrium(np.array(A),np.array(B))
    monkeypatch.setattr(research,'linprog',lambda *a,**kw:SimpleNamespace(success=True,x=np.array([1.,0.,0.,0.])))
    with pytest.raises(HTTPException,match='obedience certificate failed'):
        research.correlated_equilibrium(np.array([[0,0],[1,1]]),np.array(B))


def test_correlated_feasibility_objective_and_invalid_objective():
    a=np.zeros((2,2))
    r=research.correlated_equilibrium(a,a,objective='feasible')
    assert abs(sum(map(sum,r['joint_distribution']))-1)<1e-12
    with pytest.raises(HTTPException,match='Unsupported'):
        research.correlated_equilibrium(a,a,objective='nonsense')


@pytest.mark.parametrize('fn,a,b',[(advanced.support_equilibria,[[1,2]],[[1,2],[3,4]]),(solve_support_lp,[[float('nan')]],[[1.]]),(solve_hybrid,[[1]],[[float('inf')]])])
def test_invalid_solver_payoff_shapes(fn,a,b):
    with pytest.raises(ValueError):fn(a,b)


@pytest.mark.parametrize('kw',[dict(max_pairs=0),dict(fast_support=0)])
def test_hybrid_invalid_budget(kw):
    with pytest.raises(ValueError,match='Budgets'):solve_hybrid(A,B,**kw)


def test_analytic_solver_bad_tolerance_and_approximate_budget():
    for tol in (0,2):
        with pytest.raises(ValueError,match='tolerance'):solve_2x2(A,B,tol=tol)
    with pytest.raises(HTTPException,match='iterations'):
        approximate_nash(np.array(A),np.array(B),iterations=99)


def test_nonfinite_json_diagnostic_rejected():
    with pytest.raises(ValueError,match='Nonfinite'):json_number(np.inf)


@pytest.mark.parametrize('a,initial,steps,dt',[
 ([[1,2],[3]],None,1,.1),([[1]],[['bad']],1,.1),([[1]],[-1],1,.1),
 ([[1]],[0],1,.1),([[1]],[1],0,.1),([[1]],[1],1,.5),
 ([[1,2,3],[4,5,6]],None,1,.1)])
def test_replicator_input_constraints(a,initial,steps,dt):
    with pytest.raises(HTTPException):advanced.replicator(a,initial,steps,dt)


def test_fast_support_tolerance_merges_near_coincident_equilibria():
    # In an ill-conditioned coordination game, a genuine mixed equilibrium can
    # approach a pure one. The non-exhaustive fast solver's tolerance changes
    # how many distinct numerical representatives it reports.
    a=np.array([[1.,0.],[0.,10_000_000.]])
    fine=advanced.support_equilibria(a,a,tol=1e-12)
    coarse=advanced.support_equilibria(a,a,tol=1e-7)
    assert len(fine)==3
    assert len(coarse)==2
    assert all(abs(sum(e['row_probabilities'])-1)<1e-8 for e in fine)
    assert all(abs(sum(e['col_probabilities'])-1)<1e-8 for e in fine)
