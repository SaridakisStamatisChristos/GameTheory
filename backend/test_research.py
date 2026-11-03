import numpy as np
import pytest
from fastapi import HTTPException
from research import correlated_equilibrium, BayesianGame, bayesian_evaluate, Experiment, save_experiment, get_experiment, list_experiments

def test_correlated_pd():
    a=np.array([[3.,0.],[5.,1.]])
    b=np.array([[3.,5.],[0.,1.]])
    r=correlated_equilibrium(a,b)
    assert np.isclose(sum(map(sum,r['joint_distribution'])),1)
    assert r['maximum_obedience_violation']<1e-7
    assert np.isclose(r['joint_distribution'][1][1],1)

def test_correlated_matching_pennies():
    a=np.array([[1.,-1.],[-1.,1.]])
    r=correlated_equilibrium(a,-a)
    assert np.allclose(r['joint_distribution'],[[.25,.25],[.25,.25]],atol=1e-7)

def test_bayesian_public_state():
    g=BayesianGame(prior=[.25,.75],row_payoffs=[[[2]],[[6]]],col_payoffs=[[[3]],[[7]]],row_strategy=[0,0],col_strategy=[0,0])
    r=bayesian_evaluate(g)
    assert np.allclose(r['expected_payoffs'],[5,6])
    assert r['is_statewise_nash']

def test_bayesian_invalid_prior():
    g=BayesianGame(prior=[.2,.2],row_payoffs=[[[2]],[[6]]],col_payoffs=[[[3]],[[7]]],row_strategy=[0,0],col_strategy=[0,0])
    with pytest.raises(HTTPException):bayesian_evaluate(g)

def test_experiment_persistence(tmp_path,monkeypatch):
    monkeypatch.setenv('GAME_LAB_DB',str(tmp_path/'test.sqlite'))
    x=save_experiment(Experiment(title='demo',game={'matrix':[[1]]}))
    assert get_experiment(x['id'])['game']=={'matrix':[[1]]}
    assert list_experiments()[0]['id']==x['id']
    with pytest.raises(HTTPException) as exc:get_experiment('missing')
    assert exc.value.status_code==404
