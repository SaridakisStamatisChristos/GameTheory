"""Rejection/recall contracts for CFR, CFR+, perfect-information trees and Bayesian games."""
import math
import pytest
from fastapi import HTTPException
from cfr import CFRRequest,solve_cfr
from cfr_plus import CFRPlusRequest,solve_cfr_plus,validate_cfr_game
from extended import TreeGame,backward_induction,RepeatedGame,repeated_payoffs
from private_bayesian import PrivateBayesianGame,evaluate_private
from research import BayesianGame,bayesian_evaluate

@pytest.mark.parametrize('nodes,msg',[
 ({'root':{'player':0,'infoset':'i','actions':{'a':'missing'}}},'Missing node'),
 ({'root':{'player':0,'infoset':'i','actions':{'a':'root'}}},'cycle'),
 ({'root':{'player':0,'infoset':'i','actions':{'a':'t','b':'t'}},'t':{'payoff':0}},'shared'),
 ({'root':{'player':0,'infoset':'i'}},'actions'),
 ({'root':{'player':0,'actions':{'a':'t'}},'t':{'payoff':0}},'information'),
 ({'root':{'player':3,'infoset':'i','actions':{'a':'t'}},'t':{'payoff':0}},'Invalid player'),
 ({'root':{'player':'chance','actions':{'a':'t'},'probabilities':{'a':-1}},'t':{'payoff':0}},'chance'),
 ({'root':{'player':'chance','actions':{'a':'t'},'probabilities':{'b':1}},'t':{'payoff':0}},'chance'),
 ({'root':{'payoff':math.inf}},'payoff'),
 ({'root':{'payoff':'x'}},'payoff'),
])
def test_vanilla_tree_failures(nodes,msg):
    with pytest.raises(HTTPException,match=msg) as exc:solve_cfr(CFRRequest(nodes=nodes,iterations=2))
    assert exc.value.status_code==422


def test_vanilla_unreachable_budget_and_missing_root():
    with pytest.raises(HTTPException,match='Invalid root'):
        solve_cfr(CFRRequest(nodes={'other':{'payoff':0}},iterations=1))
    with pytest.raises(HTTPException,match='Unreachable'):
        solve_cfr(CFRRequest(nodes={'root':{'payoff':1},'orphan':{'payoff':1}},iterations=1))
    many={'root':{'player':0,'infoset':'i','actions':{str(i):f't{i}' for i in range(101)}}}
    many.update({f't{i}':{'payoff':i} for i in range(101)})
    with pytest.raises(HTTPException,match='budget'):
        solve_cfr(CFRRequest(nodes=many,iterations=20000))


def test_vanilla_infoset_owner_mismatch():
    nodes={'root':{'player':'chance','actions':{'a':'l','b':'r'},'probabilities':{'a':.5,'b':.5}},
           'l':{'player':0,'infoset':'i','actions':{'x':'lt'}},
           'r':{'player':1,'infoset':'i','actions':{'x':'rt'}},
           'lt':{'payoff':0},'rt':{'payoff':0}}
    with pytest.raises(HTTPException,match='Inconsistent information'):
        solve_cfr(CFRRequest(nodes=nodes,iterations=1))


@pytest.mark.parametrize('nodes,msg',[
 ({'root':{'player':'chance','actions':{'a':'t'},'probabilities':{'a':-1}},'t':{'payoff':1}},'chance'),
 ({'root':{'player':'chance','actions':{'a':'t'},'probabilities':{'a':.5}},'t':{'payoff':1}},'chance'),
 ({'root':{'player':0,'infoset':'i','actions':{}}},'actions'),
 ({'root':{'player':0,'infoset':'i','actions':{'':'t'}},'t':{'payoff':0}},'actions'),
 ({'root':{'player':0,'infoset':'i','actions':{'a':None}}},'actions'),
 ({'root':{'player':0,'actions':{'a':'t'}},'t':{'payoff':0}},'infoset'),
 ({'root':{'player':3,'infoset':'i','actions':{'a':'t'}},'t':{'payoff':0}},'player'),
 ({'root':{'player':0,'infoset':'i','actions':{'a':'t'}},'t':{'payoff':True}},'payoff'),
 ({'root':{'player':0,'infoset':'i','actions':{'a':'t'}},'t':{'payoff':float('nan')}},'payoff'),
 ({'root':{'player':0,'infoset':'i','actions':{'a':'root'}}},'tree'),
 ({'root':{'player':0,'infoset':'i','actions':{'a':'t','b':'t'}},'t':{'payoff':0}},'tree'),
])
def test_plus_tree_failures(nodes,msg):
    with pytest.raises(HTTPException,match=msg) as exc:validate_cfr_game(CFRPlusRequest(nodes=nodes,iterations=1))
    assert exc.value.status_code==422


def test_plus_root_unreachable_and_cross_player_infoset():
    with pytest.raises(HTTPException,match='root'):
        validate_cfr_game(CFRPlusRequest(nodes={'other':{'payoff':1}},iterations=1))
    with pytest.raises(HTTPException,match='Unreachable'):
        validate_cfr_game(CFRPlusRequest(nodes={'root':{'payoff':1},'extra':{'payoff':0}},iterations=1))
    nodes={'root':{'player':'chance','actions':{'a':'l','b':'r'},'probabilities':{'a':.5,'b':.5}},
           'l':{'player':0,'infoset':'i','actions':{'x':'lt'}},
           'r':{'player':1,'infoset':'i','actions':{'x':'rt'}},
           'lt':{'payoff':0},'rt':{'payoff':0}}
    with pytest.raises(HTTPException,match='Inconsistent infoset'):
        validate_cfr_game(CFRPlusRequest(nodes=nodes,iterations=1))


def test_plus_terminal_and_chance_game():
    terminal=solve_cfr_plus(CFRPlusRequest(nodes={'root':{'payoff':17}},iterations=1))
    assert terminal['exploitability']==0 and terminal['row_expected_payoff']==17
    assert terminal['average_strategies']=={}
    nodes={'root':{'player':'chance','actions':{'L':'left','R':'right'},'probabilities':{'L':.25,'R':.75}},
           'left':{'player':0,'infoset':'i','actions':{'a':'l1','b':'l2'}},
           'right':{'player':0,'infoset':'i','actions':{'a':'r1','b':'r2'}},
           'l1':{'payoff':4},'l2':{'payoff':0},'r1':{'payoff':0},'r2':{'payoff':2}}
    result=solve_cfr_plus(CFRPlusRequest(nodes=nodes,iterations=50))
    assert abs(sum(result['average_strategies']['i'].values())-1)<1e-12
    assert 0<=result['exploitability']<=6


@pytest.mark.parametrize('nodes,root,msg',[
 ([{'id':'root','payoffs':[0,0]},{'id':'root','payoffs':[1,1]}],'root','Duplicate'),
 ([{'id':'only','payoffs':[0,0]}],'wrong','root'),
 ([{'id':'root','player':0,'children':['t']},{'id':'t','payoffs':[1]}],'root','terminal'),
 ([{'id':'root','player':0,'children':['missing']}],'root','child'),
 ([{'id':'root','player':0,'children':['root']}],'root','Cycle'),
 ([{'id':'root','payoffs':[0,0]},{'id':'extra','payoffs':[1,1]}],'root','Unreachable'),
 ([{'id':'root','player':0,'children':[]}],'root','Nonterminal'),
 ([{'id':'root','payoffs':[0,0],'children':['root']}],'root','terminal'),
])
def test_perfect_tree_failures(nodes,root,msg):
    with pytest.raises(HTTPException,match=msg):backward_induction(TreeGame(root=root,nodes=nodes))


def test_backward_induction_shared_branch_and_tie():
    nodes=[{'id':'root','player':0,'children':['l','r']},
           {'id':'l','player':1,'children':['t']}, {'id':'r','player':1,'children':['t']},
           {'id':'t','payoffs':[2,5]}]
    result=backward_induction(TreeGame(root='root',nodes=nodes))
    assert result['path']==['root','l','t'] and result['policy']['root']['optimal_children']==['l','r']


@pytest.mark.parametrize('row,col',[([], [0]),([0], []),([1],[0]),([0],[1])])
def test_repeated_game_rejects_bad_actions(row,col):
    with pytest.raises(HTTPException,match='Invalid periodic'):
        repeated_payoffs(RepeatedGame(row_payoffs=[[1]],col_payoffs=[[2]],row_strategy=row,col_strategy=col,rounds=1))


def test_repeated_game_trace_budget_and_discount():
    r=repeated_payoffs(RepeatedGame(row_payoffs=[[1]],col_payoffs=[[2]],row_strategy=[0],col_strategy=[0],rounds=201,discount=0))
    assert r['trace']==[] and r['trace_truncated'] and r['discounted_payoffs']==[1,2]


def test_private_bayesian_zero_mass_type_and_invalid_policy():
    data=PrivateBayesianGame(joint_prior=[[1.],[0.]],row_payoffs=[[[[1],[0]]],[[[0],[1000]]]],
        col_payoffs=[[[[1],[0]]],[[[0],[0]]]],row_policy=[0,0],col_policy=[0])
    r=evaluate_private(data)
    assert r['is_bayesian_nash'] and r['row_type_conditional_deviation_gains']==[0,None]
    defaults=dict(joint_prior=[[1]],row_payoffs=[[[[1]]]],col_payoffs=[[[[1]]]],row_policy=[0],col_policy=[0])
    for change in [dict(joint_prior=[[0]]),dict(row_payoffs=[[[[1,2]]]]),dict(row_policy=[1])]:
        with pytest.raises(HTTPException):evaluate_private(PrivateBayesianGame(**{**defaults,**change}))
    from pydantic import ValidationError
    with pytest.raises(ValidationError):PrivateBayesianGame(**{**defaults,'row_payoffs':[[[['bad']]]]})
    unchecked=PrivateBayesianGame.model_construct(joint_prior=object(),row_payoffs=[[[[0]]]],
        col_payoffs=[[[[0]]]],row_policy=[0],col_policy=[0])
    with pytest.raises(HTTPException,match='Nonrectangular'):
        evaluate_private(unchecked)


def test_bayesian_public_prior_and_actions():
    data=BayesianGame(prior=[1.],row_payoffs=[[[1]]],col_payoffs=[[[2]]],row_strategy=[0],col_strategy=[0])
    assert bayesian_evaluate(data)['is_statewise_nash']
    for change in [dict(prior=[-1]),dict(row_strategy=[1]),dict(row_strategy=[]),dict(col_payoffs=[[[2,4]]])]:
        with pytest.raises(HTTPException):bayesian_evaluate(BayesianGame(**{**data.model_dump(),**change}))


def test_vanilla_cfr_boolean_player_id_is_rejected():
    nodes={'root':{'player':True,'infoset':'bad','actions':{'a':'t'}},'t':{'payoff':0}}
    with pytest.raises(HTTPException,match='Invalid player'):
        solve_cfr(CFRRequest(nodes=nodes,iterations=1))
