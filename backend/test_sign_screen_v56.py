"""v56 exact sign infeasibility certificates, negative tests, and LP oracle."""
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest

from exact_rational_lp import feasible_strict_strategy
from exact_support import ExactWorkLimit
from general_exact_search import search_exact_nash_cells


def check(a, br, support, *, screen):
    work = {'steps': 0, 'limit': 200000}
    witness = feasible_strict_strategy(
        a, br, support, work, fast_path=True, sign_screen=screen)
    return witness, work


def supports(out):
    return {(tuple(f['support']['row']), tuple(f['support']['col']))
            for f in out['families']}


def test_supported_tie_cannot_hold_at_strict_positive_mixture():
    a = [[F(2), F(2)], [F(1), F(1)], [F(0), F(0)]]
    rejected, work = check(a, (0, 1), (0, 1), screen=True)
    reference, baseline = check(a, (0, 1), (0, 1), screen=False)
    assert rejected is None and reference is None
    assert work['sign_screen_rejections'] == 1
    assert work.get('simplex_lp_calls', 0) == 0
    assert baseline['simplex_lp_calls'] == 1


def test_unsupported_action_weakly_beats_base_on_every_positive_coordinate():
    a = [[F(2), F(2)], [F(2), F(3)]]
    rejected, work = check(a, (0,), (0, 1), screen=True)
    reference, baseline = check(a, (0,), (0, 1), screen=False)
    assert rejected is None and reference is None
    assert work['sign_screen_rejections'] == 1
    assert work.get('simplex_lp_calls', 0) == 0
    assert baseline['simplex_lp_calls'] == 1


def test_screen_does_not_reject_nonuniform_feasible_tie():
    a = [[F(3), F(0)], [F(0), F(1)]]
    candidate, work = check(a, (0, 1), (0, 1), screen=True)
    baseline, _ = check(a, (0, 1), (0, 1), screen=False)
    assert candidate == baseline == [F(1, 4), F(3, 4)]
    assert work['sign_screen_operations'] == 2
    assert work['simplex_lp_calls'] == 1
    assert work.get('sign_screen_rejections', 0) == 0


def test_screen_accepts_all_tie_uniform_and_preserves_singleton_early_exit():
    z = [[F(0), F(0)], [F(0), F(0)]]
    candidate, w = check(z, (0, 1), (0, 1), screen=True)
    assert candidate == [F(1, 2), F(1, 2)]
    assert w['uniform_witness_hits'] == 1
    assert w.get('sign_screen_operations', 0) == 0
    bad = [[F(0), F(0)], [F(2), F(2)]]
    assert check(bad, (0,), (0,), screen=True)[1]['singleton_infeasible_hits'] == 1


def test_exact_sign_budget_is_charged_even_when_it_would_prove_infeasible():
    a = [[F(2), F(2)], [F(1), F(1)]]
    # Uniform probe costs 4, sign screen costs 2; stop after probe.
    w = {'steps': 0, 'limit': 5}
    with pytest.raises(ExactWorkLimit, match='sign-screen operation budget'):
        feasible_strict_strategy(a, (0, 1), (0, 1), w)
    assert w['probe_operations'] == 4
    assert w['sign_screen_operations'] == 2
    assert w.get('sign_screen_rejections', 0) == 0


def test_exact_screen_rejects_invalid_controls():
    w = {'steps': 0, 'limit': 500}
    with pytest.raises(ValueError, match='sign_screen'):
        feasible_strict_strategy([[F(0)]], (0,), (0,), w, sign_screen=1)
    with pytest.raises(ValueError, match='feasibility_sign_screen'):
        search_exact_nash_cells([[0]], [[0]], feasibility_sign_screen=0)
    with pytest.raises(ValueError, match='track_rational_bits'):
        search_exact_nash_cells([[0]], [[0]], track_rational_bits='yes')


def test_exhaustive_three_value_two_by_two_feasibility_against_simplex():
    # Compare feasibility, not a selected point in a continuum.
    for vals in product((-1, 0, 1), repeat=4):
        a = [[F(vals[0]), F(vals[1])], [F(vals[2]), F(vals[3])]]
        for br in ((0,), (1,), (0, 1)):
            for J in ((0,), (1,), (0, 1)):
                quick, _ = check(a, br, J, screen=True)
                baseline = feasible_strict_strategy(
                    a, br, J, {'steps':0,'limit':200000},
                    fast_path=False, sign_screen=False)
                assert (quick is None) == (baseline is None)
                if quick is not None:
                    assert sum(quick) == 1 and all(p > 0 for p in quick)


def test_bit_growth_diagnostics_are_opt_in_and_record_tableau_operations():
    a = [[F(3), F(0)], [F(0), F(1)]]
    w = {'steps': 0, 'limit': 200000, 'track_rational_bits': True}
    candidate = feasible_strict_strategy(a, (0,1), (0,1), w)
    assert candidate == [F(1,4), F(3,4)]
    assert w['max_tableau_numerator_bits'] >= 2
    assert w['max_tableau_denominator_bits'] >= 1
    out = search_exact_nash_cells(
        [[3,0],[0,1]], [[1,0],[0,3]], max_pairs=9, max_support=2,
        max_operations=200000, prune_supports=False, track_rational_bits=True)
    assert out['track_rational_bits'] is True
    assert out['max_tableau_numerator_bits'] is not None
    assert out['max_tableau_denominator_bits'] is not None
    off = search_exact_nash_cells(
        [[0]],[[0]],max_pairs=1,track_rational_bits=False)
    assert off['max_tableau_numerator_bits'] is None
    assert off['max_tableau_denominator_bits'] is None


def test_baseline_flag_disables_screen_and_vertices_backend_ignores_it():
    a = [[1,1],[0,0]]
    b = [[1,0],[1,0]]
    old = search_exact_nash_cells(a,b,max_pairs=9,max_support=2,
                                   feasibility_fast_path=False)
    assert old['feasibility_sign_screen'] is False
    assert old['sign_screen_operations'] == 0
    vs = search_exact_nash_cells(a,b,max_pairs=9,max_support=2,
                                  feasibility_backend='vertices')
    assert vs['feasibility_sign_screen'] is False


def test_seeded_bimatrix_exhaustive_support_differential():
    rng = np.random.default_rng(56056)
    cases = [
        (np.zeros((3,3)),np.zeros((3,3))),
        (np.asarray([[2,0],[0,1]]),np.asarray([[1,0],[0,2]])),
    ]
    for n in (2,3,4):
        for index in range(4):
            a = rng.integers(-4,5,(n,n))
            b = rng.integers(-4,5,(n,n))
            if index == 2:
                a[-1] = a[0]
                b[:, -1] = b[:, 0]
            cases.append((a,b))
    for a,b in cases:
        n = a.shape[0]
        opts = dict(max_pairs=(2**n-1)**2,max_support=n,
                    max_operations=2000000,prune_supports=False,
                    mixed_pruning=False)
        yes = search_exact_nash_cells(a,b,feasibility_sign_screen=True,**opts)
        no = search_exact_nash_cells(a,b,feasibility_sign_screen=False,**opts)
        assert yes['complete_equilibrium_set'] and no['complete_equilibrium_set']
        assert yes['support_pairs_checked'] == no['support_pairs_checked']
        assert supports(yes) == supports(no)
        assert yes['simplex_lp_calls'] <= no['simplex_lp_calls']
