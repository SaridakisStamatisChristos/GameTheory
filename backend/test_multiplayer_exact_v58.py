"""v58 exact multiplayer pure Nash and CE mathematical adversarial contracts."""
from fractions import Fraction as F
from itertools import product
import numpy as np
import pytest
from scipy.optimize import linprog
from fastapi.testclient import TestClient

import multiplayer_exact as engine
from multiplayer_exact import solve_exact_multiplayer as solve
from exact_support import ExactWorkLimit
from main import app


def coordination(players=3, actions=2):
    shape=(actions,)*players
    u=np.zeros((players,*shape),dtype=int)
    for profile in product(range(actions),repeat=players):
        if all(x==profile[0] for x in profile):
            u[(slice(None),*profile)]=2
    return u


def no_pure_match():
    matrix=np.array([[1,-1],[-1,1]],dtype=int)
    u=np.zeros((3,2,2,1),dtype=int)
    u[0,:,:,0]=matrix
    u[1,:,:,0]=-matrix
    return u


def independent_pure(u):
    shape=u.shape[1:]
    answers=set()
    for profile in product(*(range(d) for d in shape)):
        if all(u[(p,*profile)]>=u[(p,*(profile[:p]+(alt,)+profile[p+1:]))]
               for p in range(u.shape[0]) for alt in range(shape[p])):
            answers.add(profile)
    return answers


def independent_obedience_and_dual(u,out):
    shape=u.shape[1:]
    n=u.shape[0]
    profiles=list(product(*(range(s) for s in shape)))
    assert out['action_profiles']==[list(profile) for profile in profiles]
    q=[F(v) for v in out['joint_distribution_exact']]
    assert len(q)==len(profiles) and sum(q)==1 and min(q)>=0
    expected=[sum(q[k]*F(int(u[(p,*profile)]))
                  for k,profile in enumerate(profiles)) for p in range(n)]
    assert [F(v) for v in out['expected_payoffs_exact']]==expected
    rows=[[F(1)]*len(profiles),[-F(1)]*len(profiles)]
    rhs=[F(1),-F(1)]
    gains=[]
    for p,actions in enumerate(shape):
        for rec in range(actions):
            for dev in range(actions):
                if dev==rec:
                    continue
                row=[]
                for profile in profiles:
                    replacement=(*profile[:p],dev,*profile[p+1:])
                    row.append(F(int(u[(p,*replacement)]-u[(p,*profile)]))
                               if profile[p]==rec else F(0))
                rows.append(row)
                rhs.append(F(0))
                gains.append(sum(q[k]*row[k] for k in range(len(profiles))))
    assert max([F(0),*gains])==0
    assert out['obedience_inequalities_checked']==len(gains)
    assert F(out['maximum_obedience_gain_exact'])==0
    welfare=[sum(F(int(u[(p,*profile)])) for p in range(n))
             for profile in profiles]
    c=welfare if out['objective']=='welfare' else [F(0)]*len(profiles)
    dual=[F(v) for v in out['dual_multipliers_exact']]
    assert len(dual)==len(rows)
    assert min(dual)>=0
    assert all(sum(dual[i]*rows[i][j] for i in range(len(rows)))>=c[j]
               for j in range(len(profiles)))
    assert sum(rhs[i]*dual[i] for i in range(len(rhs)))==sum(
        q[j]*c[j] for j in range(len(profiles)))
    assert F(out['objective_value_exact'])==sum(q[j]*c[j]
                                                  for j in range(len(profiles)))
    return q


def independent_highs_welfare(u):
    shape=u.shape[1:]
    profiles=list(product(*(range(s) for s in shape)))
    n=u.shape[0]
    rows=[]
    for p,actions in enumerate(shape):
        for rec in range(actions):
            for dev in range(actions):
                if rec==dev:
                    continue
                rows.append([int(u[(p,*(profile[:p]+(dev,)+profile[p+1:]))] -
                                 u[(p,*profile)]) if profile[p]==rec else 0
                             for profile in profiles])
    c=[-sum(int(u[(p,*profile)]) for p in range(n))
       for profile in profiles]
    ref=linprog(c,A_ub=rows if rows else None,
                b_ub=np.zeros(len(rows)) if rows else None,
                A_eq=[np.ones(len(profiles))],b_eq=[1],
                bounds=(0,None),method='highs')
    assert ref.success,ref.message
    return -ref.fun


def test_three_player_coordination_exact_pure_and_welfare():
    u=coordination()
    out=solve(u)
    assert out['players']==3 and out['action_counts']==[2,2,2]
    assert out['profiles_checked']==8 and out['pure_nash_complete']
    assert {tuple(x['actions']) for x in out['pure_nash_equilibria']}=={
        (0,0,0),(1,1,1)}
    assert all(x['payoffs_exact']==['2','2','2']
               for x in out['pure_nash_equilibria'])
    assert out['status']=='certified_exact_multiplayer_correlated'
    assert out['objective_value_exact']=='6'
    assert not out['mixed_nash_complete']
    assert out['correlated_equilibrium_computed']
    assert not out['correlated_equilibrium_set_complete']
    independent_obedience_and_dual(u,out)


def test_no_pure_nash_but_exact_multiplayer_correlated_exists():
    u=no_pure_match()
    for objective in ('welfare','feasible'):
        out=solve(u,objective=objective)
        assert out['pure_nash_equilibria']==[]
        q=independent_obedience_and_dual(u,out)
        assert q==[F(1,4)]*4
        assert out['objective_value_exact']=='0'
        assert out['action_counts']==[2,2,1]


def test_zero_ties_four_player_and_five_player_singletons():
    u=np.zeros((4,2,2,2,2),dtype=int)
    out=solve(u)
    assert len(out['pure_nash_equilibria'])==16
    independent_obedience_and_dual(u,out)
    singleton=np.asarray([1,2,-3,4,6],dtype=int).reshape((5,1,1,1,1,1))
    s=solve(singleton)
    assert len(s['pure_nash_equilibria'])==1
    assert s['expected_payoffs_exact']==['1','2','-3','4','6']
    assert s['objective_value_exact']=='10'
    independent_obedience_and_dual(singleton,s)


def test_pure_only_mode_is_complete_without_mixed_claims():
    u=no_pure_match()
    out=solve(u,mode='pure',max_operations=1)
    assert out['status']=='certified_complete_pure_nash'
    assert out['pure_nash_equilibria']==[]
    assert out['pure_nash_complete']
    assert not out['correlated_equilibrium_computed']
    assert not out['mixed_nash_complete']
    assert 'dual_multipliers_exact' not in out
    assert 'joint_distribution_exact' not in out


def test_seeded_multiplayer_exact_vs_independent_pure_and_highs():
    rng=np.random.default_rng(58058)
    shapes=[(2,2,2),(2,2,3),(2,3,2),(3,2,2),(2,2,2,2),
            (1,2,2,2,2)]
    for shape in shapes:
        for sample in range(3):
            u=rng.integers(-4,5,size=(len(shape),*shape))
            if sample==1:
                u[0]=u[0].flat[0]
            out=solve(u,max_operations=2000000)
            exact={tuple(e['actions']) for e in out['pure_nash_equilibria']}
            assert exact==independent_pure(u)
            independent_obedience_and_dual(u,out)
            assert abs(float(F(out['objective_value_exact'])) -
                       independent_highs_welfare(u))<1e-8


def test_multiplayer_exact_rational_decimal_model():
    u=coordination()
    a=u.astype(float)
    a[0,0,0,0]=.1
    out=solve(a,mode='pure')
    assert out['input_model'].startswith('decimal strings')
    assert len(out['pure_nash_equilibria'])==2
    assert out['pure_nash_equilibria'][0]['payoffs_exact'][0]=='1/10'


@pytest.mark.parametrize('bad,fragment',[
    (np.zeros((2,2,2)), '3 to 5'),
    (np.zeros((6,1,1,1,1,1,1)), '3 to 5'),
    (np.zeros((3,2,2)), 'matching finite'),
    (np.zeros((3,5,1,1)), 'matching finite'),
    (np.zeros((3,4,4,4)), 'matching finite'),
    (np.full((3,1,1,1),float('nan')), 'matching finite'),
    (np.full((3,1,1,1),float('inf')), 'matching finite'),
    (np.full((3,1,1,1),1000001.), 'matching finite'),
    ([['bad']], 'numeric rectangular'),
    ([[1], [1,2], [3]], 'numeric rectangular'),
])
def test_invalid_multiplayer_payoff_models(bad,fragment):
    with pytest.raises(ValueError,match=fragment):
        solve(bad)


@pytest.mark.parametrize('arguments,fragment',[
    ({'mode':'unknown'}, 'mode'),
    ({'objective':'unknown'}, 'objective'),
    ({'max_operations':0}, 'budget'),
    ({'max_operations':True}, 'budget'),
    ({'max_operations':2000001}, 'budget'),
    ({'max_operations':1.5}, 'budget'),
])
def test_multiplayer_option_validation(arguments,fragment):
    with pytest.raises(ValueError,match=fragment):
        solve(coordination(),**arguments)


def test_rational_work_budget_fails_closed():
    with pytest.raises(ExactWorkLimit):
        solve(coordination(),max_operations=1)


def test_adversarial_bad_primal_and_obedience_fails_closed(monkeypatch):
    class Broken:
        candidate=None
        def __init__(self,*unused):
            pass
        def solve(self):
            return self.candidate
    for candidate,part in [
        (None,'probability simplex'),
        ([F(0)]*4,'probability simplex'),
        ([F(2),F(-1),F(0),F(0)], 'probability simplex'),
        ([F(1),F(0),F(0),F(0)], 'obedience'),
    ]:
        Broken.candidate=candidate
        monkeypatch.setattr(engine,'_RationalSimplex',Broken)
        with pytest.raises(AssertionError,match=part):
            solve(no_pure_match())


def test_adversarial_bad_dual_fails_closed(monkeypatch):
    class Broken:
        calls=0
        dual=None
        def __init__(self,*unused):
            pass
        def solve(self):
            Broken.calls+=1
            return [F(1)] if Broken.calls%2 else self.dual
    u=np.ones((3,1,1,1),dtype=int)
    for candidate,part in [
        (None,'dual LP'),
        ([F(-1),F(0)], 'primal-dual'),
        ([F(0)], 'primal-dual'),
        ([F(0),F(0)], 'primal-dual'),
    ]:
        Broken.calls=0
        Broken.dual=candidate
        monkeypatch.setattr(engine,'_RationalSimplex',Broken)
        with pytest.raises(AssertionError,match=part):
            solve(u)


def test_multiplayer_api_http_and_existing_routes_unchanged():
    with TestClient(app,raise_server_exceptions=False) as client:
        r=client.post('/api/multiplayer/exact',json={
            'payoffs':coordination().tolist(), 'max_operations':1000000})
        assert r.status_code==200,r.text
        assert r.json()['objective_value_exact']=='6'
        p=client.post('/api/multiplayer/exact',json={
            'payoffs':no_pure_match().tolist(),'mode':'pure',
            'max_operations':1})
        assert p.status_code==200,p.text
        assert p.json()['pure_nash_equilibria']==[]
        assert client.post('/api/multiplayer/exact',json={
            'payoffs':coordination().tolist(),'max_operations':1}).status_code==422
        assert client.post('/api/multiplayer/exact',json={
            'payoffs':coordination().tolist(),'mode':'invalid'}).status_code==422
        assert client.post('/api/multiplayer/exact',json={
            'payoffs':[[[1,2]],[[1,2]]]}).status_code==422
        legacy=client.post('/api/correlated-equilibrium/exact',json={
            'row_payoffs':[[1,-1],[-1,1]],
            'col_payoffs':[[-1,1],[1,-1]]})
        assert legacy.status_code==200,legacy.text
        assert legacy.json()['maximum_obedience_violation_exact']=='0'
