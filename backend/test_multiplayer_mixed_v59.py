"""v59 independent N-player mixed Nash incentive audits and failure contracts."""
from fractions import Fraction as F
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

import multiplayer_mixed as engine
from multiplayer_mixed import solve_multiplayer_mixed as solve


def matching_with_dummy():
    m = np.asarray([[1, -1], [-1, 1]])
    u = np.zeros((3, 2, 2, 1), dtype=int)
    u[0, :, :, 0] = m
    u[1, :, :, 0] = -m
    return u


def parity_game():
    u = np.zeros((3, 2, 2, 2), dtype=int)
    for p in range(3):
        for profile in product(range(2), repeat=3):
            u[(p, *profile)] = (1 if profile[p] ==
                (profile[(p+1) % 3] ^ profile[(p+2) % 3]) else -1)
    return u


def dominant_game(players=3):
    u = np.zeros((players, *(2 for _ in range(players))), dtype=int)
    for p in range(players):
        for profile in product(range(2), repeat=players):
            u[(p, *profile)] = int(profile[p] == 0)
    return u


def independently_verify_exact_epsilon(payoffs, candidate):
    """Direct product-profile rational audit, independent of root equations."""
    n = payoffs.shape[0]
    shape = payoffs.shape[1:]
    strategies = [[F(x) for x in probs] for probs in candidate['strategies_exact']]
    assert len(strategies) == n
    assert all(sum(x) == 1 and all(v >= 0 for v in x) for x in strategies)
    assert all({a for a,x in enumerate(probs) if x > 0} == set(candidate['support'][p])
               for p,probs in enumerate(strategies))
    expected = []
    regrets = []
    for p in range(n):
        values = []
        for action in range(shape[p]):
            value = F(0)
            for profile in product(*(range(k) for k in shape)):
                if profile[p] != action:
                    continue
                multiplier = F(1)
                for q in range(n):
                    if q != p:
                        multiplier *= strategies[q][profile[q]]
                value += multiplier * F(str(float(payoffs[(p,*profile)])))
            values.append(value)
        payoff = sum(strategies[p][a]*values[a] for a in range(shape[p]))
        regrets.append(max(values)-payoff)
        expected.append(payoff)
    epsilon = max(regrets)
    assert [F(x) for x in candidate['player_unilateral_regrets_exact']] == regrets
    assert [F(x) for x in candidate['expected_payoffs_exact']] == expected
    assert F(candidate['epsilon_exact']) == epsilon
    assert epsilon >= 0
    assert candidate['status'] == ('certified_exact_rational_nash' if epsilon == 0
                                   else 'verified_exact_epsilon_nash')
    return epsilon


def test_three_player_no_pure_matching_has_exact_mixed_nash():
    u=matching_with_dummy()
    out=solve(u)
    assert out['pure_nash_complete'] and not out['pure_nash_equilibria']
    assert out['mixed_nash_complete'] is False
    assert out['all_support_patterns_attempted']
    assert out['support_patterns_total']==9
    assert out['support_patterns_examined']==9
    assert out['mixed_candidates_found'] >= 1
    target=[c for c in out['mixed_candidates'] if c['support']==[[0,1],[0,1],[0]]]
    assert len(target)==1
    assert target[0]['strategies_exact'] == [['1/2','1/2'],['1/2','1/2'],['1']]
    assert independently_verify_exact_epsilon(u,target[0])==0
    assert out['incomplete_reason'] is None


def test_genuinely_three_player_parity_game_mixed_certificates():
    u=parity_game()
    out=solve(u,starts_per_pattern=2,max_candidates=64)
    assert out['mixed_candidates_found'] > 0
    assert out['all_support_patterns_attempted']
    assert any(c['support']==[[0,1],[0,1],[0,1]] for c in out['mixed_candidates'])
    for candidate in out['mixed_candidates']:
        assert independently_verify_exact_epsilon(u,candidate) <= F('1e-8')


def test_four_player_and_five_player_scope():
    four=np.zeros((4,2,2,2,2))
    result=solve(four,starts_per_pattern=1,max_candidates=128)
    assert result['support_patterns_total']==81
    assert result['all_support_patterns_attempted']
    assert len(result['pure_nash_equilibria'])==16
    assert any(c['support']==[[0,1]]*4 for c in result['mixed_candidates'])
    for c in result['mixed_candidates']:
        assert independently_verify_exact_epsilon(four,c)==0
    five=dominant_game(5)
    partial=solve(five,max_support_patterns=18,starts_per_pattern=1)
    assert partial['players']==5
    assert partial['support_patterns_total']==243
    assert partial['incomplete_reason']=='support_pattern_budget'
    assert partial['support_patterns_examined']==18
    assert not partial['all_support_patterns_attempted']
    assert len(partial['pure_nash_equilibria'])==1


def test_genuine_no_mixed_candidate_not_claimed_complete():
    out=solve(dominant_game(),starts_per_pattern=1)
    assert out['status']=='no_mixed_candidates_verified'
    assert out['mixed_candidates_found']==0
    assert out['all_support_patterns_attempted']
    assert not out['mixed_nash_complete']
    assert out['pure_nash_equilibria'][0]['actions']==[0,0,0]


def test_partial_root_evaluation_budget_explicit():
    out=solve(matching_with_dummy(),max_evaluations=1)
    assert out['incomplete_reason']=='function_evaluation_budget'
    assert out['function_evaluations']==1
    assert not out['all_support_patterns_attempted']
    assert not out['mixed_nash_complete']


def test_candidate_budget_exposes_truncation():
    out=solve(np.zeros((3,2,2,2)),max_candidates=1,starts_per_pattern=1)
    assert out['mixed_candidates_found']==1
    assert out['incomplete_reason']=='candidate_budget'
    assert not out['all_support_patterns_attempted']


def test_seed_determinism_and_bounded_work():
    u=parity_game()
    kwargs=dict(seed=753,starts_per_pattern=3,max_evaluations=2000)
    a=solve(u,**kwargs)
    b=solve(u,**kwargs)
    assert a['mixed_candidates']==b['mixed_candidates']
    assert a['root_calls']==b['root_calls']
    assert a['function_evaluations']<=2000
    assert a['function_evaluations']==b['function_evaluations']


def test_perturbed_numerical_root_has_exact_positive_epsilon_certificate(monkeypatch):
    original=engine.root
    def perturbed(fn,z,**options):
        result=original(fn,z,**options)
        if len(z)==2 and np.max(np.abs(result.x-.5))<1e-7:
            result.x=np.array([.500000001,.499999999])
        return result
    monkeypatch.setattr(engine,'root',perturbed)
    out=solve(matching_with_dummy(),tolerance=1e-6,starts_per_pattern=1)
    assert any(F(c['epsilon_exact'])>0 for c in out['mixed_candidates'])
    for c in out['mixed_candidates']:
        assert independently_verify_exact_epsilon(matching_with_dummy(),c)<=F('1e-6')


def test_nonfinite_roots_are_never_certified(monkeypatch):
    monkeypatch.setattr(engine,'root',lambda *args,**kwargs:
                        SimpleNamespace(x=np.array([float('nan')])))
    out=solve(matching_with_dummy(),starts_per_pattern=1)
    assert out['mixed_candidates_found']==0
    assert not out['mixed_nash_complete']


def test_invalid_outside_simplex_roots_are_rejected(monkeypatch):
    monkeypatch.setattr(engine,'root',lambda fn,z,**kwargs:
                        SimpleNamespace(x=np.full(len(z),-2.0)))
    out=solve(matching_with_dummy(),starts_per_pattern=1)
    assert out['mixed_candidates_found']==0


@pytest.mark.parametrize('kwargs,phrase',[
    ({'tolerance':-1},'tolerance'),({'tolerance':1},'tolerance'),
    ({'tolerance':float('nan')},'tolerance'),({'tolerance':True},'tolerance'),
    ({'seed':-1},'seed'),({'seed':True},'seed'),({'seed':2**31},'seed'),
    ({'max_support_patterns':0},'max_support_patterns'),
    ({'max_support_patterns':1025},'max_support_patterns'),
    ({'max_support_patterns':True},'max_support_patterns'),
    ({'starts_per_pattern':0},'starts_per_pattern'),
    ({'starts_per_pattern':13},'starts_per_pattern'),
    ({'starts_per_pattern':True},'starts_per_pattern'),
    ({'max_evaluations':0},'max_evaluations'),
    ({'max_evaluations':100001},'max_evaluations'),
    ({'max_evaluations':True},'max_evaluations'),
    ({'max_candidates':0},'max_candidates'),
    ({'max_candidates':129},'max_candidates'),
    ({'max_candidates':True},'max_candidates'),
])
def test_invalid_search_controls_fail_closed(kwargs,phrase):
    with pytest.raises(ValueError,match=phrase):
        solve(matching_with_dummy(),**kwargs)
