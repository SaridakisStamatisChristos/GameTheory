"""v45 general engine: independent game oracles, budgets, and route contracts."""
from fractions import Fraction
import numpy as np
import pytest
from general_nash import (
    _validate, eliminate_strict_pure_dominance, _zero_sum_candidate,
    solve_general_nash)
from general_exact_search import search_exact_nash_cells
from exact_nash_set import solve_exact_nash_set
from test_exact_nash_set_v43 import is_nash, membership


def test_exact_rational_classification_matches_reference_small_games():
    rng=np.random.default_rng(4520)
    for _ in range(18):
        a=rng.integers(-2,3,size=(2,2))
        b=rng.integers(-2,3,size=(2,2))
        result=search_exact_nash_cells(a,b,max_pairs=9,max_support=2)
        oracle=solve_exact_nash_set(a,b)
        assert result['status']=='complete'
        assert result['complete_equilibrium_set'] is True
        assert result['support_pairs_checked']==9
        supports={tuple(map(tuple,(f['support']['row'],f['support']['col'])))
                  for f in result['families']}
        reference={tuple(map(tuple,(f['support']['row'],f['support']['col'])))
                   for f in oracle['equilibrium_families']}
        assert supports==reference
        for f in result['families']:
            x=tuple(Fraction(t) for t in f['row_probabilities_exact'])
            y=tuple(Fraction(t) for t in f['col_probabilities_exact'])
            assert is_nash(a,b,x,y)
            assert membership(oracle['equilibrium_families'],x,y)==1


def test_continuum_zero_payoffs_exact_vs_budgets():
    z=np.zeros((2,2))
    complete=solve_general_nash(z,z,mode='exact',max_exact_pairs=9,max_exact_support=2)
    assert complete['status']=='complete_exact_set'
    assert complete['equilibrium_set_complete'] is True
    assert len(complete['exact_search']['families'])==9
    assert complete['best_candidate']['certificate']['valid']
    for opts,reason in [({'max_exact_pairs':2},'pair_budget'),
                        ({'max_exact_support':1},'support_size_cap'),
                        ({'max_exact_operations':1},'operation_budget')]:
        r=solve_general_nash(z,z,mode='exact',**opts)
        assert r['status']=='partial_exact_search'
        assert not r['equilibrium_set_complete']
        assert r['exact_search']['incomplete_reason']==reason


def test_general_engine_handles_large_dimensions_without_core_size_gates():
    for m,n in ((8,12),(30,50)):
        a=np.zeros((m,n));b=np.zeros((m,n))
        r=solve_general_nash(a,b,mode='auto',max_pure_cells=5000)
        assert r['original_shape']==[m,n]
        assert r['reduced_shape']==[m,n]
        assert r['pure_search_exhausted']
        assert r['status']=='certified_candidate'
        assert r['best_candidate']['evidence']=='exact_pure_equilibrium'
        assert r['best_candidate']['certificate']['valid']
        assert r['equilibrium_set_complete'] is False


def test_iterated_dominance_preserves_original_game_equilibrium():
    a=np.array([[0,0],[2,2.]])
    b=np.array([[3,1],[3,1.]])
    rows,cols,steps=eliminate_strict_pure_dominance(a,b)
    assert rows==[1] and cols==[0]
    assert len(steps)==2
    r=solve_general_nash(a,b,mode='exact')
    assert r['reduced_shape']==[1,1]
    assert r['equilibrium_set_complete']
    assert r['best_candidate']['row_probabilities']==[0.,1.]
    assert r['best_candidate']['col_probabilities']==[1.,0.]
    assert r['best_candidate']['certificate']['valid']
    assert len(r['strict_dominance_eliminations'])==2


def test_matching_pennies_numerical_zero_sum_route_and_certificate():
    a=np.array([[1.,-1.],[-1.,1.]])
    b=-a
    r=solve_general_nash(a,b)
    c=r['best_candidate']
    assert c['method']=='constant_sum_minimax_lp'
    assert c['certificate']['valid']
    assert c['normalized_minimax_duality_gap']<1e-8
    assert np.allclose(c['row_probabilities'],[0.5,0.5])
    assert np.allclose(c['col_probabilities'],[0.5,0.5])
    assert not r['equilibrium_set_complete']


def test_constant_sum_rectangular_minimax_matches_scipy_bound():
    rng=np.random.default_rng(4521)
    a=rng.integers(-10,11,size=(8,12)).astype(float)
    b=3.0-a
    r=solve_general_nash(a,b,max_pure_cells=1)
    assert r['best_candidate'] is not None
    assert r['pure_search_exhausted'] is False
    c=r['best_candidate']
    assert c['method']=='constant_sum_minimax_lp'
    assert c['certificate']['valid']
    x=np.asarray(c['row_probabilities']);y=np.asarray(c['col_probabilities'])
    assert np.max(a@y)-np.min(x@a)<1e-6


def test_general_sum_numerical_lh_and_approximate_routes():
    a=np.array([[3.,0.],[0.,2.]])
    b=np.array([[2.,0.],[0.,3.]])
    r=solve_general_nash(a,b,max_pure_cells=1)
    assert r['best_candidate'] is not None
    assert r['best_candidate']['certificate']['valid']
    assert r['best_candidate']['method']=='independent_lemke_howson'
    v=solve_general_nash(a,b,mode='approximate',iterations=100,restarts=1,seed=42)
    assert v['best_candidate'] is not None
    assert v['best_candidate']['evidence']=='approximate_candidate'
    assert v['best_candidate']['iterations_used']<=100
    assert not v['equilibrium_set_complete']


def test_medium_general_sum_falls_back_to_bounded_approximate():
    n=12
    a=np.eye(n)*3
    b=-np.eye(n)*3+np.fromfunction(lambda i,j:(i+j)%3,(n,n))
    r=solve_general_nash(a,b,mode='auto',iterations=100,restarts=1,seed=2)
    assert r['best_candidate'] is not None
    assert r['best_candidate']['method']=='multi_start_fictitious_play'
    assert r['best_candidate']['certificate']['simplex_error']<1e-9
    assert r['equilibrium_set_complete'] is False


def test_rectangular_partial_search_no_false_completeness_claim():
    a=np.zeros((6,7));b=np.zeros((6,7))
    r=search_exact_nash_cells(a,b,max_pairs=5,max_support=1)
    assert r['support_pairs_total']==(2**6-1)*(2**7-1)
    assert r['support_pairs_checked']==5
    assert r['status']=='partial' and not r['complete_equilibrium_set']
    assert all(is_nash(a,b,
          tuple(Fraction(z) for z in f['row_probabilities_exact']),
          tuple(Fraction(z) for z in f['col_probabilities_exact']))
          for f in r['families'])


@pytest.mark.parametrize('a,b',[
    ([],[]),
    ([[1,2]],[[1]]),
    ([[float('inf')]],[[1]]),
    ([[float('nan')]],[[1]]),
    ([[1],[]],[[1],[]]),
    ([['hello']],[[1]]),
])
def test_invalid_matrices_fail(a,b):
    with pytest.raises(ValueError):
        solve_general_nash(a,b)


@pytest.mark.parametrize('opts',[
    {'mode':'wrong'}, {'iterations':0}, {'restarts':0},
    {'iterations':200000,'restarts':64}, {'seed':-1},
    {'max_pure_cells':0}, {'max_pure_cells':True},
    {'iterations':True}, {'restarts':True},
])
def test_invalid_portfolio_budgets_fail(opts):
    with pytest.raises(ValueError):
        solve_general_nash([[1]],[[1]],**opts)


@pytest.mark.parametrize('opts',[
    {'max_pairs':0}, {'max_pairs':4097}, {'max_pairs':True},
    {'max_operations':0}, {'max_operations':2000001},
    {'max_support':0}, {'max_support':21}, {'max_support':False},
])
def test_invalid_exact_budgets_fail(opts):
    with pytest.raises(ValueError):
        search_exact_nash_cells([[1]],[[1]],**opts)


def test_faulted_exact_verifier_never_returns_fabricated_certification(monkeypatch):
    import general_exact_search as ex
    monkeypatch.setattr(ex,'_verify_exact_equilibrium',lambda *args:False)
    with pytest.raises(AssertionError,match='failed Nash'):
        search_exact_nash_cells([[1]],[[1]])


def test_fault_injected_invalid_heuristic_is_rejected(monkeypatch):
    import general_nash as gn
    monkeypatch.setattr(gn,'approximate_nash',lambda *a,**kw:{
        'profile':{'row_probabilities':[-1.], 'col_probabilities':[1.],
                   'iteration':100}})
    r=solve_general_nash([[1]],[[1]],mode='approximate')
    assert not r['candidates']
    assert r['status']=='inconclusive'
    assert not r['candidate_certified_within_tolerance']


def test_numerical_solver_failure_falls_back_honestly(monkeypatch):
    import general_nash as gn
    monkeypatch.setattr(gn,'_zero_sum_candidate',lambda *a:None)
    monkeypatch.setattr(gn,'solve_independent_lh',
                        lambda *a,**kw:{'equilibria':[]})
    r=gn.solve_general_nash([[1,-1],[-1,1]],[[-1,1],[1,-1]],
                            mode='auto',iterations=100,restarts=1)
    assert r['best_candidate']['method']=='multi_start_fictitious_play'
    assert not r['equilibrium_set_complete']

def test_independent_scipy_lp_support_oracle_3x3():
    import itertools
    from test_exact_support_v42 import reference_support_feasible
    rng=np.random.default_rng(4547)
    for _ in range(8):
        a=rng.integers(-3,4,size=(3,3))
        b=rng.integers(-3,4,size=(3,3))
        r=search_exact_nash_cells(a,b,max_pairs=49,max_support=3,
                                  max_operations=200000)
        assert r['complete_equilibrium_set']
        observed={(tuple(f['support']['row']),tuple(f['support']['col']))
                  for f in r['families']}
        expected=set()
        for k in range(1,4):
            for I in itertools.combinations(range(3),k):
                for l in range(1,4):
                    for J in itertools.combinations(range(3),l):
                        if reference_support_feasible(a,b,I,J):
                            expected.add((I,J))
        assert observed==expected,(a.tolist(),b.tolist(),observed^expected)


def test_invalid_first_lp_certificate_does_not_block_better_fallback(monkeypatch):
    import general_nash as gn
    # Fabricate a non-equilibrium minimax output to check that the
    # independent certificate cannot be bypassed by a successful LP flag.
    monkeypatch.setattr(gn,'_zero_sum_candidate',
                        lambda *args:(np.array([1.,0.]),np.array([1.,0.]),
                                      {'normalized_minimax_duality_gap':0.}))
    a=[[1,-1],[-1,1]]
    b=[[-1,1],[1,-1]]
    r=gn.solve_general_nash(a,b,mode='auto')
    assert len(r['candidates'])>=2
    assert r['candidates'][0]['certificate']['valid'] is False
    assert r['candidate_certified_within_tolerance']
    assert r['best_candidate']['certificate']['valid']


def test_indeterminate_lp_and_pivot_allow_bounded_approximate_fallback(monkeypatch):
    import general_nash as gn
    monkeypatch.setattr(gn,'_zero_sum_candidate',
                        lambda *args:(np.array([1.,0.]),np.array([1.,0.]),{}))
    monkeypatch.setattr(gn,'solve_independent_lh',
                        lambda *args,**kw:{'equilibria':[]})
    r=gn.solve_general_nash([[1,-1],[-1,1]],[[-1,1],[1,-1]],
                            iterations=100,restarts=1)
    assert len(r['candidates'])==2
    assert r['candidates'][1]['method']=='multi_start_fictitious_play'
    assert not r['equilibrium_set_complete']


def test_simple_constant_sum_degenerate_case_and_lp_failures(monkeypatch):
    import general_nash as gn
    a=np.zeros((2,2));b=np.zeros((2,2))
    x=gn._zero_sum_candidate(a,b)
    assert x is not None and x[2]['normalized_minimax_duality_gap']<1e-9
    class Failed:
        success=False
    monkeypatch.setattr(gn,'linprog',lambda *args,**kw:Failed())
    assert gn._zero_sum_candidate(a,b) is None
