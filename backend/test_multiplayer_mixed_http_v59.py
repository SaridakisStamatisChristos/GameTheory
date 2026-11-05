"""v59 multiplayer mixed endpoint validation, exact epsilon, and isolation tests."""
from fractions import Fraction as F
import numpy as np
from fastapi.testclient import TestClient
from main import app


def sample():
    a=np.array([[1,-1],[-1,1]])
    u=np.zeros((3,2,2,1),dtype=int)
    u[0,:,:,0]=a
    u[1,:,:,0]=-a
    return {'payoffs':u.tolist()}


def post(data):
    with TestClient(app,raise_server_exceptions=False) as client:
        return client.post('/api/multiplayer/mixed',json=data)


def test_http_three_player_matching_returns_exact_rational_certificate():
    r=post(sample())
    assert r.status_code==200,r.text
    d=r.json()
    assert d['mixed_candidates_found']>=1
    assert not d['pure_nash_equilibria']
    assert d['pure_nash_complete'] is True
    assert d['mixed_nash_complete'] is False
    found=[x for x in d['mixed_candidates']
           if x['support']==[[0,1],[0,1],[0]]]
    assert len(found)==1
    assert found[0]['status']=='certified_exact_rational_nash'
    assert F(found[0]['epsilon_exact'])==0


def test_http_partial_budget_is_explicit_not_complete():
    r=post({**sample(),'max_evaluations':1})
    assert r.status_code==200,r.text
    d=r.json()
    assert d['incomplete_reason']=='function_evaluation_budget'
    assert d['function_evaluations']==1
    assert not d['mixed_nash_complete']


def test_http_rejects_malformed_game_and_options():
    base=sample()
    for opts in ({'seed':-1},{'starts_per_pattern':0},
                 {'max_evaluations':0},{'max_candidates':0},
                 {'tolerance':0},{'max_support_patterns':1025}):
        r=post({**base,**opts})
        assert r.status_code==422,(opts,r.text)
    for bad in ({'payoffs':[[],[],[]]},
                {'payoffs':base['payoffs'][:2]},
                {'payoffs':[[[1,2]],[[1,2]],[[1,2]]]}):
        r=post(bad)
        assert r.status_code==422,(bad,r.text)


def test_http_v58_exact_multiplayer_unchanged():
    with TestClient(app,raise_server_exceptions=False) as client:
        r=client.post('/api/multiplayer/exact',json={
            'payoffs':sample()['payoffs'],'mode':'pure'})
    assert r.status_code==200,r.text
    assert r.json()['pure_nash_complete'] is True
