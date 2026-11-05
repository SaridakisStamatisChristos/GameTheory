"""Independent stochastic zero-sum games, security conditions and failure gates."""
import numpy as np
import pytest
from scipy.linalg import solve
import stochastic_zero_sum as sg
from stochastic_zero_sum import solve_discounted_zero_sum_stochastic as stochastic


def single(reward, discount=0.9, **options):
    return stochastic([[[reward]]], [[[[1.]]]], discount=discount, **options)


def test_one_state_one_action_closed_form_and_stationary_policies():
    out = single(2., discount=.75, max_iterations=200)
    assert out['status'] == 'converged_numerical'
    assert abs(out['value_estimates'][0]-8) <= out['value_error_bound']+1e-11
    assert out['value_error_bound'] <= 1e-6
    assert out['row_stationary_policy'] == [[1.]]
    assert out['col_stationary_policy'] == [[1.]]
    assert out['stationary_policy_regret_bound'] == 2*out['value_error_bound']
    assert out['state_saddle_gaps'] == [0]
    assert out['transition_max_row_sum_correction'] == 0


def test_one_state_matching_pennies_exact_matrix_reference():
    r = [[[1,-1],[-1,1]]]
    out = stochastic(r,np.ones((1,2,2,1)),discount=.9,max_iterations=4)
    assert out['tolerance_met'] and out['iterations'] == 0
    assert out['value_estimates'] == [0]
    assert out['row_stationary_policy'][0] == pytest.approx([.5,.5])
    assert out['col_stationary_policy'][0] == pytest.approx([.5,.5])
    assert out['max_bellman_residual_bound'] < 1e-9


def test_all_tie_game_and_zero_discount_solves_directly():
    out = stochastic(np.full((1,2,3),4.),np.ones((1,2,3,1)),
                     discount=0.,max_iterations=1)
    assert out['tolerance_met']
    assert out['value_estimates'] == pytest.approx([4.])
    assert out['row_stationary_policy'] == [[1,0]]
    assert out['col_stationary_policy'] == [[1,0,0]]


def test_independent_markov_reward_linear_system():
    p=np.array([[.8,.2],[.3,.7]])
    r=np.array([2.,-3.])
    out=stochastic(r[:,None,None],p[:,None,None,:],
                   discount=.8,tolerance=1e-7,max_iterations=250)
    expected=solve(np.eye(2)-.8*p,r)
    assert out['tolerance_met']
    assert np.max(np.abs(expected-out['value_estimates'])) <= out['value_error_bound']+1e-10


def test_action_independent_transition_two_state_closed_form():
    r=np.array([[[1,-1],[-1,1]],[[2,0],[0,1]]],dtype=float)
    p=np.array([[.6,.4],[.2,.8]])
    t=np.broadcast_to(p[:,None,None,:],(2,2,2,2)).copy()
    out=stochastic(r,t,discount=.6,tolerance=1e-7,max_iterations=250)
    expected=solve(np.eye(2)-.6*p,np.array([0,2/3]))
    assert out['tolerance_met']
    assert np.max(np.abs(expected-out['value_estimates'])) <= out['value_error_bound']+1e-8
    assert out['row_stationary_policy'][0] == pytest.approx([.5,.5],abs=1e-7)
    assert out['col_stationary_policy'][0] == pytest.approx([.5,.5],abs=1e-7)
    assert out['row_stationary_policy'][1] == pytest.approx([1/3,2/3],abs=1e-7)
    assert out['col_stationary_policy'][1] == pytest.approx([1/3,2/3],abs=1e-7)


def test_seeded_action_dependent_transition_certificates():
    rng=np.random.default_rng(57057)
    r=rng.integers(-4,5,(3,2,2)).astype(float)
    p=rng.dirichlet([2,2,2],size=(3,2,2))
    out=stochastic(r,p,discount=.5,tolerance=1e-6,max_iterations=100)
    assert out['tolerance_met']
    v=np.asarray(out['value_estimates'])
    for s in range(3):
        x=np.asarray(out['row_stationary_policy'][s])
        y=np.asarray(out['col_stationary_policy'][s])
        assert min(x.min(),y.min()) >= 0
        assert x.sum() == pytest.approx(1)
        assert y.sum() == pytest.approx(1)
        q=r[s]+.5*np.einsum('ijb,b->ij',p[s],v)
        low=min(x@q); high=max(q@y)
        assert low == pytest.approx(out['row_security_values'][s],abs=1e-9)
        assert high == pytest.approx(out['column_ceiling_values'][s],abs=1e-9)
        assert max(abs(low-v[s]),abs(high-v[s])) <= out['max_bellman_residual_bound']+1e-10
        assert high-low < 1e-7


def test_iteration_budget_is_partial_not_false_convergence():
    out=single(1.,discount=.95,tolerance=1e-10,max_iterations=1)
    assert out['status']=='partial_bound'
    assert not out['tolerance_met']
    assert out['value_error_bound'] > out['requested_value_error_tolerance']
    assert out['iterations'] == 1
    assert out['stage_games_solved'] == 2
    assert out['value_estimates'] == pytest.approx([1.])


def test_tiny_transition_row_error_is_reported_and_normalized():
    p=np.array([[[[.3,.7+2e-13]]],[[[.4,.6]]]])
    out=stochastic(np.zeros((2,1,1)),p,discount=0.,max_iterations=1)
    assert out['tolerance_met'] and out['transition_max_row_sum_correction']>0


@pytest.mark.parametrize('reward,transition,message',[
    ([[[1,2]]], [[[[1]]]], 'Transition shape'),
    ([[[1,np.inf]]], np.ones((1,1,2,1)), 'finite rewards'),
    ([[[0]]], [[[[-.1]]]], 'Transition probabilities'),
    ([[[0]]], [[[[.8]]]], 'Transition probabilities'),
    ([[[1e7]]], [[[[1]]]], 'reward exceeds'),
    ([[[0]]]*9, np.ones((9,1,1,9))/9, 'Transition shape'),
    ([[[0]*7]], np.ones((1,1,7,1)), 'Transition shape'),
    ([[[0],[0,1]]], [], 'numeric rectangular'),
    ([[['junk']]], [[[[1]]]], 'numeric rectangular'),
])
def test_malformed_stochastic_models_fail_closed(reward,transition,message):
    with pytest.raises(ValueError,match=message):
        stochastic(reward,transition)


@pytest.mark.parametrize('options,message',[
    ({'discount':-1}, 'Discount'),
    ({'discount':1}, 'Discount'),
    ({'discount':float('nan')}, 'Discount'),
    ({'discount':True}, 'Discount'),
    ({'tolerance':0}, 'Tolerance'),
    ({'tolerance':float('nan')}, 'Tolerance'),
    ({'tolerance':True}, 'Tolerance'),
    ({'max_iterations':0}, 'iteration'),
    ({'max_iterations':501}, 'iteration'),
    ({'max_iterations':True}, 'iteration'),
])
def test_bad_iteration_controls_fail_closed(options,message):
    with pytest.raises(ValueError,match=message):
        single(1,**options)


def test_combined_iteration_state_budget_is_enforced():
    r=np.zeros((8,1,1))
    p=np.eye(8)[:,None,None,:]
    with pytest.raises(ValueError,match='iteration'):
        stochastic(r,p,max_iterations=400)


def test_stage_inconclusive_lp_does_not_return_evidence(monkeypatch):
    class Fail:
        success=False
    monkeypatch.setattr(sg,'linprog',lambda *args,**kwargs:Fail())
    with pytest.raises(ValueError,match='inconclusive'):
        stochastic([[[1,0],[0,1]]],np.ones((1,2,2,1)))


def test_stage_corrupt_probability_fails_closed(monkeypatch):
    class Fake:
        success=True
        x=np.array([0.,0.,0.])
    monkeypatch.setattr(sg,'linprog',lambda *args,**kwargs:Fake())
    with pytest.raises(ValueError,match='invalid probabilities'):
        stochastic([[[1,0],[0,1]]],np.ones((1,2,2,1)))


def test_stage_bad_saddle_certificate_fails_closed(monkeypatch):
    class Fake:
        success=True
        x=np.array([1.,0.,0.])
    monkeypatch.setattr(sg,'linprog',lambda *args,**kwargs:Fake())
    with pytest.raises(ValueError,match='saddle certificate'):
        stochastic([[[1,0],[0,1]]],np.ones((1,2,2,1)))


def test_stage_nonfinite_residual_fails_closed(monkeypatch):
    monkeypatch.setattr(sg,'_matrix_saddle',
                        lambda _: (np.ones(1),np.ones(1),float('inf'),float('inf')))
    with pytest.raises(ValueError,match='Residual bound overflowed'):
        single(0)
