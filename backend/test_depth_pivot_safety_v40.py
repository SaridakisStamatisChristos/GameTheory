"""Recursion containment and corrupted complementary-pivot robustness."""
import sys
import numpy as np
import pytest
from fastapi import HTTPException
from cfr import CFRRequest,solve_cfr
from cfr_plus import CFRPlusRequest,validate_cfr_game
from advanced import replicator
import independent_lh


def chance_chain(n):
    d={f'n{i}':{'player':'chance','actions':{'go':f'n{i+1}'},'probabilities':{'go':1.}} for i in range(n)}
    d[f'n{n}']={'payoff':0}
    return d


@pytest.mark.parametrize('solver',['vanilla','plus'])
def test_recursion_error_is_bounded_validation_result(solver):
    nodes=chance_chain(500)
    old=sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(450)
        with pytest.raises(HTTPException,match='recursion depth'):
            if solver=='vanilla':solve_cfr(CFRRequest(nodes=nodes,root='n0',iterations=1))
            else:validate_cfr_game(CFRPlusRequest(nodes=nodes,root='n0',iterations=1))
    finally:sys.setrecursionlimit(old)


def test_vanilla_cfr_chance_tree_is_exact_expectation():
    nodes={'root':{'player':'chance','actions':{'a':'t1','b':'t2'},'probabilities':{'a':.25,'b':.75}},
           't1':{'payoff':4},'t2':{'payoff':0}}
    r=solve_cfr(CFRRequest(nodes=nodes,iterations=3))
    assert r['average_strategies']=={} and r['row_expected_payoff']==1


def test_independent_pivot_cycle_rejected(monkeypatch):
    monkeypatch.setattr(independent_lh,'_pivot',lambda *a,**kw:1)
    r=independent_lh.solve_independent_lh([[1,-1],[-1,1]],[[-1,1],[1,-1]],max_labels=1,max_pivots=20)
    assert not r['equilibria'] and any('cycle' in f['error'] for f in r['failures'])


def test_independent_pivot_zero_strategy_mass_rejected(monkeypatch):
    monkeypatch.setattr(independent_lh,'_pivot',lambda mat,basis,slack,entering:entering)
    r=independent_lh.solve_independent_lh([[1,-1],[-1,1]],[[-1,1],[1,-1]],max_labels=1,max_pivots=3)
    assert not r['equilibria'] and any('Zero strategy mass' in f['error'] for f in r['failures'])


def test_independent_pivot_negative_strategy_mass_rejected(monkeypatch):
    counter={'n':0}
    def corrupt_pivot(mat,basis,slack,entering):
        counter['n']+=1
        if counter['n']==1:
            basis[:]=[0,1];return 3
        basis[:]=[2,3];return 0
    monkeypatch.setattr(independent_lh,'_pivot',corrupt_pivot)
    real_solve=independent_lh.np.linalg.solve
    def corrupt_solve(*a,**kw):
        if counter['n']>=2:
            counter['n']+=1
            return np.array([-1.,2.]) if counter['n']==3 else np.array([.5,.5])
        return real_solve(*a,**kw)
    monkeypatch.setattr(independent_lh.np.linalg,'solve',corrupt_solve)
    r=independent_lh.solve_independent_lh([[1,-1],[-1,1]],[[-1,1],[1,-1]],max_labels=1)
    assert not r['equilibria'] and any('Negative strategy mass' in f['error'] for f in r['failures'])
