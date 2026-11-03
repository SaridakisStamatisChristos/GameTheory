import pytest
from fastapi import HTTPException
from cfr import CFRRequest,solve_cfr

def matching_pennies():
    nodes={'root':{'player':0,'infoset':'row','actions':{'H':'cH','T':'cT'}}}
    for r in 'HT':
        nodes['c'+r]={'player':1,'infoset':'col','actions':{c:'t'+r+c for c in 'HT'}}
        for c in 'HT':nodes['t'+r+c]={'payoff':1 if r==c else -1}
    return nodes

def test_matching_pennies():
    result=solve_cfr(CFRRequest(nodes=matching_pennies(),iterations=4000))
    for mix in result['average_strategies'].values():
        assert all(abs(p-.5)<.04 for p in mix.values())
    assert abs(result['row_expected_payoff'])<.05

def test_cycle():
    with pytest.raises(HTTPException):solve_cfr(CFRRequest(nodes={'root':{'player':0,'infoset':'x','actions':{'a':'root'}}}))

def test_invalid_chance():
    with pytest.raises(HTTPException):solve_cfr(CFRRequest(nodes={'root':{'player':'chance','actions':{'a':'t'},'probabilities':{'a':.5}},'t':{'payoff':1}}))
