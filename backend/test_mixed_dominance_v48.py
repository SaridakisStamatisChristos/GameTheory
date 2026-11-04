"""v48: falsify mixed-dominance pruning with independent LP and Nash oracles."""
from fractions import Fraction
from itertools import combinations
import numpy as np
import pytest
from scipy.optimize import linprog
from fastapi.testclient import TestClient
from mixed_dominance_pruning import ConditionalMixedDominancePruner
from general_exact_search import search_exact_nash_cells
from general_nash import solve_general_nash
from exact_support import ExactWorkLimit
from main import app


def _oracle_mixed_dominance(matrix, target, supported_opponents):
    """Independent SciPy/HiGHS LP: may *other* pure actions weakly dominate target?"""
    a=np.asarray(matrix,dtype=float)
    alternatives=[k for k in range(len(a)) if k != target]
    if not alternatives:
        return False
    js=list(supported_opponents)
    optimized=linprog(
        c=-np.sum(a[np.ix_(alternatives,js)],axis=1),
        A_ub=-a[np.ix_(alternatives,js)].T,
        b_ub=-a[target,js],
        A_eq=np.ones((1,len(alternatives))),
        b_eq=np.array([1.0]),
        bounds=(0,None),method='highs')
    if not optimized.success:
        return False
    return bool(-optimized.fun-np.sum(a[target,js]) > 1e-8)


def _verify_certificate(a,b,certificate,support_rows,support_cols):
    p=certificate['player']
    matrix=[[Fraction(str(float(v))) for v in row] for row in
            (np.asarray(a) if p=='row' else np.asarray(b).T)]
    support=support_cols if p=='row' else support_rows
    target=certificate['dominated_action']
    mixture=certificate['dominating_mixture_exact']
    weights=[Fraction(item['probability_exact']) for item in mixture]
    assert sum(weights)==1 and all(w>0 for w in weights)
    assert target not in [item['action'] for item in mixture]
    gains={j:sum(w*matrix[item['action']][j] for w,item in zip(weights,mixture))
           -matrix[target][j] for j in support}
    assert all(g>=0 for g in gains.values())
    assert {j for j,g in gains.items() if g>0}==set(certificate['strict_gain_actions'])
    assert any(g>0 for g in gains.values())


def _supports(out):
    return {(tuple(x['support']['row']),tuple(x['support']['col'])) for x in out['families']}


def test_mixed_dominance_detects_convex_combinations_beyond_pure_dominance():
    a=[[0,0],[2,-1],[-1,2]]
    b=[[0,0],[0,0],[0,0]]
    p=ConditionalMixedDominancePruner(a,b)
    w={'steps':0,'limit':100000}
    cert=p.explain((0,),(0,1),w)
    assert cert and cert['player']=='row' and cert['dominance_kind']=='mixed'
    _verify_certificate(a,b,cert,(0,),(0,1))
    assert p.lp_attempts>=1
    previous=p.lp_attempts
    assert p.explain((0,),(0,1),w)==cert
    assert p.cache_hits>=1 and p.lp_attempts==previous
    # A single alternative is insufficient: mixture is strictly stronger.
    assert all(not all(a[k][j]>=a[0][j] for j in (0,1)) for k in (1,2))


def test_column_mixture_certificate_and_exact_tie_boundary():
    a=[[0,0,0],[0,0,0]]
    b=[[0,2,-1],[0,-1,2]]
    pruner=ConditionalMixedDominancePruner(a,b)
    cert=pruner.explain((0,1),(0,),{'steps':0,'limit':100000})
    assert cert and cert['player']=='col'
    _verify_certificate(a,b,cert,(0,1),(0,))
    # Equality on all opponent actions is not strict mixed dominance.
    flat=[[0,0],[1,-1],[-1,1]]
    p=ConditionalMixedDominancePruner(flat,[[0,0]]*3)
    assert p.explain((0,),(0,1),{'steps':0,'limit':100000}) is None


def test_exact_mixed_dominance_vs_independent_scipy_oracle_randomized():
    rng=np.random.default_rng(4811)
    for m,n,reps in ((2,3,12),(3,3,20),(4,3,8)):
        for _ in range(reps):
            a=rng.integers(-4,5,size=(m,n))
            b=rng.integers(-4,5,size=(m,n))
            p=ConditionalMixedDominancePruner(a,b)
            work={'steps':0,'limit':1000000}
            for size in range(1,n+1):
                for J in combinations(range(n),size):
                    for i in range(m):
                        cert=p._dominating_mixture(0,i,J,work)
                        assert (cert is not None)==_oracle_mixed_dominance(a,i,J)
                        if cert: _verify_certificate(a,b,cert,(i,),J)
            for size in range(1,m+1):
                for I in combinations(range(m),size):
                    for j in range(n):
                        cert=p._dominating_mixture(1,j,I,work)
                        assert (cert is not None)==_oracle_mixed_dominance(b.T,j,I)
                        if cert: _verify_certificate(a,b,cert,I,(j,))


def test_complete_sets_unchanged_across_none_pure_and_mixed_pruning():
    rng=np.random.default_rng(4802)
    cases=[([[0,0],[2,-1],[-1,2]],[[0,0]]*3),
           ([[0,0],[1,-1],[-1,1]],[[0,0]]*3)]
    for _ in range(9):
        cases.append((rng.integers(-3,4,size=(3,3)).tolist(),
                      rng.integers(-3,4,size=(3,3)).tolist()))
    for a,b in cases:
        opts={'max_pairs':225,'max_support':3,'max_operations':1000000}
        ref=search_exact_nash_cells(a,b,prune_supports=False,mixed_pruning=False,**opts)
        pure=search_exact_nash_cells(a,b,prune_supports=True,mixed_pruning=False,**opts)
        mixed=search_exact_nash_cells(a,b,prune_supports=True,mixed_pruning=True,**opts)
        assert ref['complete_equilibrium_set']
        assert ref['support_pairs_checked']==pure['support_pairs_checked']==mixed['support_pairs_checked']
        assert _supports(ref)==_supports(pure)==_supports(mixed)
        assert mixed['dominance_pruned_pairs']>=pure['dominance_pruned_pairs']
        assert mixed['dominance_pruned_pairs']==mixed['mixed_dominance_pruned_pairs']+(
            pure['dominance_pruned_pairs'])
        assert mixed['lp_pairs_attempted']+mixed['dominance_pruned_pairs']==mixed['support_pairs_checked']
        for c in mixed['pruning_witnesses_sample']:
            if c['dominance_kind']=='mixed':
                _verify_certificate(a,b,c,c['support']['row'],c['support']['col'])


def test_mixed_pruning_fails_closed_when_rational_work_budget_exhausts():
    a=[[0,0],[2,-1],[-1,2]]
    b=[[0,0]]*3
    p=ConditionalMixedDominancePruner(a,b)
    with pytest.raises(ExactWorkLimit):
        p.explain((0,),(0,),{'steps':0,'limit':1})
    r=search_exact_nash_cells(a,b,prune_supports=False,mixed_pruning=True,
                              max_pairs=21,max_operations=1)
    assert not r['complete_equilibrium_set']
    assert r['status']=='partial' and r['incomplete_reason']=='operation_budget'
    assert r['support_pairs_checked']==0


def test_mixed_pruning_validation_and_api_is_opt_in():
    for cls in (ConditionalMixedDominancePruner,):
        with pytest.raises(ValueError):cls([[0]],[[float('nan')]])
        with pytest.raises(ValueError):cls([[0,1]],[[0]])
    with pytest.raises(ValueError):
        search_exact_nash_cells([[0]],[[0]],mixed_pruning='true')
    with pytest.raises(ValueError):
        solve_general_nash([[0]],[[0]],mixed_pruning=1)
    for active in (False,True):
        r=solve_general_nash([[0,0],[2,-1],[-1,2]],[[0,0]]*3,
                            mode='exact',max_exact_pairs=21,
                            mixed_pruning=active,max_exact_support=3,
                            max_exact_operations=300000)
        assert r['exact_search']['mixed_pruning_enabled'] is active
        assert r['equilibrium_set_complete']
    with TestClient(app,raise_server_exceptions=False) as client:
        payload={'row_payoffs':[[0,0],[2,-1],[-1,2]],
                 'col_payoffs':[[0,0]]*3,'mode':'exact',
                 'max_exact_pairs':21,'max_exact_support':3,
                 'max_exact_operations':300000}
        response=client.post('/api/equilibria/general',json={**payload,'mixed_pruning':True})
        assert response.status_code==200,response.text
        assert response.json()['exact_search']['mixed_pruning_enabled']
        assert client.post('/api/equilibria/general',
                           json={**payload,'mixed_pruning':'maybe'}).status_code==422


def test_pure_scan_exhaustion_is_not_claimed_after_first_found_equilibrium():
    a=b=[[0,0],[0,0]]
    r=solve_general_nash(a,b,mode='auto')
    assert r['pure_cells_checked']==1
    assert r['pure_search_exhausted'] is False
    # Smallest game has exactly one cell, so examining it does exhaust search.
    one=solve_general_nash([[1]],[[1]],mode='auto')
    assert one['pure_cells_checked']==1 and one['pure_search_exhausted'] is True
    skipped=solve_general_nash(a,b,mode='auto',max_pure_cells=1)
    assert skipped['pure_cells_checked']==0 and skipped['pure_search_exhausted'] is False
