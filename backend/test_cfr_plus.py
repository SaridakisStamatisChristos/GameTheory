import pytest
from fastapi import HTTPException
from cfr_plus import CFRPlusRequest,solve_cfr_plus

def pennies():
    n={'root':{'player':0,'infoset':'row','actions':{'H':'h','T':'t'}}}
    for r,k in [('H','h'),('T','t')]:
        n[k]={'player':1,'infoset':'col','actions':{'H':k+'h','T':k+'t'}}
        for c in 'HT':n[k+c.lower()]={'payoff':1 if r==c else -1}
    return n

def test_cfr_plus_pennies():
    r=solve_cfr_plus(CFRPlusRequest(nodes=pennies(),iterations=2000))
    assert r['certificate_status']=='exact_perfect_recall_tree_best_response'
    assert r['exploitability']<.06
    assert abs(r['row_expected_payoff'])<.06

def test_perfect_recall_rejected():
    n=pennies();n['h']['player']=0;n['t']['player']=0;n['h']['infoset']='again';n['t']['infoset']='again'
    with pytest.raises(HTTPException):solve_cfr_plus(CFRPlusRequest(nodes=n,iterations=1))

def test_chance_and_bound():
    n={'root':{'player':'chance','actions':{'a':'a','b':'b'},'probabilities':{'a':.25,'b':.75}},'a':{'payoff':4},'b':{'payoff':0}}
    r=solve_cfr_plus(CFRPlusRequest(nodes=n,iterations=2))
    assert r['row_expected_payoff']==1
    assert r['exploitability']==0

def test_best_response_matches_bruteforce_small_game():
    from best_response import best_response
    from itertools import product
    nodes=pennies()
    profile={'row':{'H':.3,'T':.7},'col':{'H':.8,'T':.2}}
    for p in (0,1):
        v, decisions=best_response(nodes,'root',profile,p)
        candidates=[]
        for a in 'HT':
            pol={i:dict(x) for i,x in profile.items()}
            pol['row' if p==0 else 'col']={x:float(x==a) for x in 'HT'}
            def ev(k):
                n=nodes[k]
                if 'payoff' in n:return n['payoff']
                return sum(pol[n['infoset']][a]*ev(c) for a,c in n['actions'].items())
            candidates.append(ev('root'))
        assert v==pytest.approx(max(candidates) if p==0 else min(candidates))


def test_large_strategy_space_no_profile_enumeration():
    nodes={'root':{'player':'chance','actions':{},'probabilities':{}}}
    for j in range(15):
        k=f'n{j}'
        nodes['root']['actions'][str(j)]=k
        nodes['root']['probabilities'][str(j)]=1/15
        nodes[k]={'player':0,'infoset':f'i{j}','actions':{'A':f'a{j}','B':f'b{j}'}}
        nodes[f'a{j}']={'payoff':1}
        nodes[f'b{j}']={'payoff':0}
    r=solve_cfr_plus(CFRPlusRequest(nodes=nodes,iterations=5,max_br_profiles=1))
    assert r['certificate_status']=='exact_perfect_recall_tree_best_response'
    assert r['exploitability']>=0
