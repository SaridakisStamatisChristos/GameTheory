"""Regression tests for numerical invariants and API error contracts.

Each case exercises a mathematical property or an input boundary independently
of the normal success-case fixtures. Run with `python -m pytest -q`.
"""
import numpy as np
from fastapi.testclient import TestClient
from main import app

client = TestClient(app, raise_server_exceptions=False)

def post(path, payload):
    return client.post(path, json=payload)

def test_ce_obedience_in_original_payoff_units():
    a = np.array([[0., 0.], [1., 1.]])
    b = np.array([[1e15, 0.], [0., 1e15]])
    response = post('/api/correlated-equilibrium', {'row_payoffs': a.tolist(), 'col_payoffs': b.tolist()})
    assert response.status_code == 200
    q = np.asarray(response.json()['joint_distribution'])
    unscaled_gain = float(q[0] @ (a[1] - a[0]))
    assert unscaled_gain <= 1e-7, f'False CE certificate: unilateral gain {unscaled_gain}'

def test_support_lp_must_find_existing_interior_equilibrium():
    a = [[1., -1.], [-1., 1.]]
    b = [[0., 1e5], [0., -1e-5]]
    response = post('/api/equilibria/support-lp', {'row_payoffs': a, 'col_payoffs': b})
    assert response.status_code == 200
    # Unique exact interior equilibrium exists with x_0 ~ 1e-10 and y_0 = 1/2.
    equilibria = response.json()['equilibria']
    assert any(0 < eq['row_probabilities'][0] < 1 and 0 < eq['col_probabilities'][0] < 1 for eq in equilibria), response.json()

def test_strict_dominance_does_not_include_only_weak_dominance():
    body = {'row_payoffs': [[0, 0], [0, 1]], 'col_payoffs': [[0, 0], [0, 0]]}
    response = post('/api/analyze', body)
    assert response.status_code == 200
    assert response.json()['strictly_dominated_by_pure']['row'] == []

def test_replicator_preserves_zero_support():
    body = {'payoffs': [[0., 0.], [1e8, 0.]], 'initial': [1., 0.], 'steps': 1, 'dt': .05}
    response = post('/api/evolution', body)
    assert response.status_code == 200
    assert response.json()['final_distribution'][1] == 0.

def test_bayesian_nash_checks_rare_positive_probability_types():
    body = {'joint_prior': [[1-1e-12], [1e-12]],
            'row_payoffs': [[[[0.], [0.]]], [[[0.], [100.]]]],
            'col_payoffs': [[[[0.], [0.]]], [[[0.], [0.]]]],
            'row_policy': [0, 0], 'col_policy': [0]}
    response = post('/api/bayesian/private-evaluate', body)
    assert response.status_code == 200
    assert response.json()['is_bayesian_nash'] is False

def test_matching_pennies_equilibria_scale_invariant():
    s=1e-13
    a=[[s,-s],[-s,s]]
    b=[[-v for v in row] for row in a]
    response=post('/api/analyze', {'row_payoffs':a,'col_payoffs':b})
    assert response.status_code==200
    result=response.json()
    assert result['pure_nash']==[]
    assert len(result['interior_mixed_nash_2x2'])>=1

def test_backward_induction_selects_actual_maximizing_action():
    body = {'root':'root','players':2,'nodes':[
        {'id':'root','player':0,'children':['bad','good']},
        {'id':'bad','payoffs':[0.,0.]},
        {'id':'good','payoffs':[1e-11,0.]}]}
    response=post('/api/tree/backward-induction', body)
    assert response.status_code==200
    assert response.json()['policy']['root']['chosen']=='good'

def test_vanilla_cfr_rejects_dag_aliasing_for_tree_contract():
    nodes={'n0':{'player':'chance','actions':{'a':'n1','b':'n1'},'probabilities':{'a':.5,'b':.5}},
           'n1':{'payoff':1}}
    response=post('/api/imperfect-information/cfr', {'root':'n0','nodes':nodes,'iterations':1})
    assert response.status_code==422

def test_valid_large_translation_zero_sum_is_not_500():
    s=1e12
    a=[[1+s,-1+s],[-1+s,1+s]]
    response=post('/api/zero-sum', {'payoffs':a})
    assert response.status_code==200, response.text

def test_finite_extreme_approximate_request_is_not_500():
    a=[[1e308,-1e308],[-1e308,1e308]]
    b=[[-v for v in row] for row in a]
    response=post('/api/equilibria/approximate',{'row_payoffs':a,'col_payoffs':b,'iterations':100,'restarts':1})
    assert response.status_code!=500, response.text
