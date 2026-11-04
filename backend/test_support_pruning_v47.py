"""v47 sound conditional dominance pruning: exact equivalence and falsifiers."""
from fractions import Fraction
from itertools import product
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import app
from general_exact_search import search_exact_nash_cells
from general_nash import solve_general_nash
from support_dominance_pruning import ConditionalDominancePruner
from test_exact_nash_set_v43 import is_nash


def _supports(result):
    return {(tuple(f['support']['row']), tuple(f['support']['col']))
            for f in result['families']}


def _audit(a, b, max_pairs=4096, max_support=20):
    opts={'max_pairs':max_pairs,'max_support':max_support,
          'max_operations':1000000}
    optimized=search_exact_nash_cells(a,b,prune_supports=True,**opts)
    baseline=search_exact_nash_cells(a,b,prune_supports=False,**opts)
    assert optimized['support_pairs_checked']==baseline['support_pairs_checked']
    assert optimized['support_pairs_total']==baseline['support_pairs_total']
    assert optimized['complete_equilibrium_set']==baseline['complete_equilibrium_set']
    assert optimized['incomplete_reason']==baseline['incomplete_reason']
    assert _supports(optimized)==_supports(baseline)
    assert optimized['dominance_pruned_pairs'] >= 0
    assert optimized['lp_pairs_attempted']+optimized['dominance_pruned_pairs']==optimized['support_pairs_checked']
    assert baseline['lp_pairs_attempted']==baseline['support_pairs_checked']
    assert optimized['exact_operations'] <= baseline['exact_operations']
    assert len(optimized['pruning_witnesses_sample'])<=12
    for entry in optimized['families']:
        x=tuple(Fraction(z) for z in entry['row_probabilities_exact'])
        y=tuple(Fraction(z) for z in entry['col_probabilities_exact'])
        assert is_nash(np.asarray(a).tolist(),np.asarray(b).tolist(),x,y)
    return optimized


def test_conditional_weak_dominance_requires_strict_advantage_inside_support():
    a=[[2,1],[1,1]]
    b=[[0,0],[0,0]]
    f=ConditionalDominancePruner(a,b)
    assert f.explain((1,),(0,1))=={'player':'row','dominated_action':1,'dominates_action':0}
    assert f.explain((1,),(1,)) is None
    # Repeated calls exercise the cache, without changing exclusion witnesses.
    assert f.explain((1,),(0,1))==f.explain((1,),(0,1))
    assert f.explain((0,),(0,1)) is None


def test_symmetric_column_dominance_and_boundary_kink():
    a=[[0,0],[0,0]]
    b=[[2,1],[1,1]]
    p=ConditionalDominancePruner(a,b)
    assert p.explain((0,1),(1,))=={'player':'col','dominated_action':1,'dominates_action':0}
    assert p.explain((1,),(1,)) is None
    assert p.explain((0,1),(1,))==p.explain((0,1),(1,))
    assert p.explain((0,),(0,)) is None


def test_identical_payoffs_dont_prune_ties_or_boundary_equilibria():
    a=b=[[0,0,0],[0,0,0],[0,0,0]]
    p=ConditionalDominancePruner(a,b)
    for I in ((0,),(1,2),(0,1,2)):
        for J in ((0,),(0,2),(0,1,2)):
            assert p.explain(I,J) is None
    r=_audit(a,b)
    assert r['complete_equilibrium_set']
    assert r['feasible_cells_found']==49
    assert r['dominance_pruned_pairs']==0


def test_exhaustive_two_by_two_binary_payoffs():
    for flat in product((-1,1),repeat=8):
        a=np.asarray(flat[:4]).reshape(2,2)
        b=np.asarray(flat[4:]).reshape(2,2)
        r=_audit(a,b)
        assert r['complete_equilibrium_set']
        assert r['support_pairs_checked']==9


def test_seeded_degenerate_rectangular_and_larger_games():
    rng=np.random.default_rng(47201)
    for m,n,repeat in ((2,3,12),(3,3,20),(3,4,10),(4,4,6),(6,7,3)):
        for _ in range(repeat):
            a=rng.integers(-3,4,size=(m,n))
            b=rng.integers(-3,4,size=(m,n))
            r=_audit(a,b,max_pairs=120 if m>=6 else 4096,
                     max_support=2 if m>=6 else 20)
            assert r['support_pairs_checked']>0


def test_dominance_heavy_games_skip_lp_without_false_completeness():
    m=n=4
    a=np.array([[4-i]*n for i in range(m)])
    b=np.array([[4-j for j in range(n)] for _ in range(m)])
    r=_audit(a,b)
    assert r['complete_equilibrium_set']
    assert r['feasible_cells_found']==1
    assert r['dominance_pruned_pairs']>100
    assert len(r['pruning_witnesses_sample'])==12
    assert r['exact_operations']>0
    for cert in r['pruning_witnesses_sample']:
        I=cert['support']['row'];J=cert['support']['col']
        p=cert['dominated_action'];q=cert['dominates_action']
        matrix=a if cert['player']=='row' else b.T
        opponent=J if cert['player']=='row' else I
        assert all(matrix[q,j]>=matrix[p,j] for j in opponent)
        assert any(matrix[q,j]>matrix[p,j] for j in opponent)


def test_partial_budget_cannot_gain_false_completeness_from_pruning():
    m=n=6
    a=np.array([[m-i]*n for i in range(m)])
    b=np.array([[n-j for j in range(n)] for _ in range(m)])
    r=_audit(a,b,max_pairs=25,max_support=1)
    assert not r['complete_equilibrium_set']
    assert r['incomplete_reason']=='pair_budget'
    assert r['support_pairs_checked']==25
    assert r['lp_pairs_attempted']<25


def test_invalid_payload_and_invalid_pruning_type_rejected():
    with pytest.raises(ValueError):
        ConditionalDominancePruner([[1,2]],[[1]])
    with pytest.raises(ValueError):
        ConditionalDominancePruner([[1,float('inf')]],[[0,0]])
    with pytest.raises(ValueError):
        search_exact_nash_cells([[0]],[[0]],prune_supports=1)
    with pytest.raises(ValueError):
        solve_general_nash([[0]],[[0]],prune_supports='no')


def test_general_portfolio_and_http_opt_out_contract():
    a=[[3,2],[2,1]]
    b=[[3,2],[2,1]]
    for pruning in (True,False):
        r=solve_general_nash(a,b,mode='exact',max_exact_pairs=9,
                             max_exact_support=2,prune_supports=pruning)
        assert r['equilibrium_set_complete']
        assert r['exact_search']['support_pruning_enabled'] is pruning
        assert r['best_candidate']['certificate']['valid']
    with TestClient(app,raise_server_exceptions=False) as client:
        # Matching pennies has no globally dominated pure actions, but
        # conditional dominance can still reject impossible strict supports.
        data={'row_payoffs':[[1,-1],[-1,1]],
              'col_payoffs':[[-1,1],[1,-1]],
              'mode':'exact','max_exact_pairs':9,'max_exact_support':2}
        yes=client.post('/api/equilibria/general',json={**data,'prune_supports':True})
        no=client.post('/api/equilibria/general',json={**data,'prune_supports':False})
        assert yes.status_code==no.status_code==200
        assert yes.json()['exact_search']['dominance_pruned_pairs']>0
        assert no.json()['exact_search']['dominance_pruned_pairs']==0
        assert client.post('/api/equilibria/general',json={**data,'prune_supports':2}).status_code==422
