"""v55 exact certified shortcut: differential proof obligations and budget safety."""
from fractions import Fraction as F
from itertools import combinations, product
import numpy as np
import pytest
from exact_rational_lp import _RationalSimplex, feasible_strict_strategy
from exact_support import ExactWorkLimit
from general_exact_search import search_exact_nash_cells
from test_exact_nash_set_v43 import is_nash


def _check_single(a, best, support, *, fast):
    w = {'steps': 0, 'limit': 100000}
    result = feasible_strict_strategy(a, best, support, w, fast_path=fast)
    return result, w


def test_exhaustive_binary_two_action_payoff_supports_match_original_lp():
    # All 2^4 payoff matrices, prescribed BR subsets, and all opponent
    # supports. No floating tolerance; original LP serves as oracle.
    for v in product((-1, 1), repeat=4):
        a = [[F(v[0]), F(v[1])], [F(v[2]), F(v[3])]]
        for best in ((0,), (1,), (0, 1)):
            for support in ((0,), (1,), (0, 1)):
                quick, _ = _check_single(a, best, support, fast=True)
                baseline, _ = _check_single(a, best, support, fast=False)
                assert (quick is None) == (baseline is None)
                if quick is not None:
                    assert sum(quick) == 1 and all(z > 0 for z in quick)


def test_singleton_is_complete_uniform_candidate_with_exact_deviation():
    # Asymmetric prescribed best response: singleton can fail on strict
    # improvement or on an equality demanded by the support.
    a = [[F(2), F(2)], [F(1), F(2)], [F(1), F(1)]]
    good, work = _check_single(a, (0,), (0,), fast=True)
    assert good == [F(1)]
    assert work['uniform_witness_hits'] == 1
    assert work.get('simplex_lp_calls', 0) == 0
    bad, work = _check_single(a, (1,), (0,), fast=True)
    assert bad is None and work['singleton_infeasible_hits'] == 1
    tie_bad, work = _check_single(a, (0, 1), (0,), fast=True)
    assert tie_bad is None and work['singleton_infeasible_hits'] == 1


def test_uniform_feasible_fraction_witness_and_nonuniform_fallback():
    zero = [[F('0.1'), F('0.1')], [F('0.1'), F('0.1')]]
    out, w = _check_single(zero, (0, 1), (0, 1), fast=True)
    assert out == [F(1, 2), F(1, 2)]
    assert w['uniform_witness_hits'] == 1 and w.get('simplex_lp_calls', 0) == 0
    # Unique nonuniform mixture with no valid uniform witness.
    matching = [[F(3), F(0)], [F(0), F(1)]]
    out, w = _check_single(matching, (0, 1), (0, 1), fast=True)
    assert out == [F(1, 4), F(3, 4)]
    assert w['simplex_lp_calls'] == 1
    assert w.get('uniform_witness_hits', 0) == 0


def test_probe_cost_is_never_free_and_invalid_flag_is_rejected():
    a = [[F(0), F(0)], [F(0), F(0)]]
    with pytest.raises(ExactWorkLimit, match='probe operation budget'):
        feasible_strict_strategy(a, (0,), (0,), {'steps':0, 'limit':1})
    with pytest.raises(ExactWorkLimit):
        feasible_strict_strategy(a, (0,), (0, 1), {'steps':0, 'limit':3})
    with pytest.raises(ValueError, match='fast_path'):
        feasible_strict_strategy(a, (0,), (0,), {'steps':0, 'limit':1}, fast_path=1)
    with pytest.raises(ValueError, match='feasibility_fast_path'):
        search_exact_nash_cells([[0]], [[0]], feasibility_fast_path='yes')


def _supports(r):
    return {(tuple(f['support']['row']), tuple(f['support']['col']))
            for f in r['families']}


def test_complete_degenerate_nash_family_equivalence_with_baseline():
    z = np.zeros((3, 3))
    opts = dict(max_pairs=49, max_support=3, max_operations=200000,
                prune_supports=False, mixed_pruning=False)
    quick = search_exact_nash_cells(z, z, feasibility_fast_path=True, **opts)
    baseline = search_exact_nash_cells(z, z, feasibility_fast_path=False, **opts)
    assert quick['complete_equilibrium_set'] and baseline['complete_equilibrium_set']
    assert _supports(quick) == _supports(baseline)
    assert quick['feasible_cells_found'] == 49
    assert quick['simplex_lp_calls'] == 0
    assert baseline['simplex_lp_calls'] > 0
    assert quick['uniform_witness_hits'] > 0
    assert quick['exact_operations'] < baseline['exact_operations']
    for item in quick['families']:
        x = [F(t) for t in item['row_probabilities_exact']]
        y = [F(t) for t in item['col_probabilities_exact']]
        assert is_nash(z.tolist(), z.tolist(), x, y)


def test_seeded_random_degenerate_and_nonuniform_differential():
    rng = np.random.default_rng(55055)
    for size, loops in ((2, 10), (3, 8), (4, 4)):
        for case in range(loops):
            a = rng.integers(-3, 4, size=(size, size)).astype(float)
            b = rng.integers(-3, 4, size=(size, size)).astype(float)
            if case % 3 == 0:
                a[:, -1] = a[:, 0]
                b[:, -1] = b[:, 0]
            opts = dict(max_pairs=(2**size-1)**2, max_support=size,
                        max_operations=2000000, prune_supports=True,
                        mixed_pruning=False)
            quick = search_exact_nash_cells(a, b, feasibility_fast_path=True, **opts)
            baseline = search_exact_nash_cells(a, b, feasibility_fast_path=False, **opts)
            assert quick['complete_equilibrium_set'] and baseline['complete_equilibrium_set']
            assert _supports(quick) == _supports(baseline)
            assert quick['support_pairs_checked'] == baseline['support_pairs_checked']
            for item in quick['families']:
                x = [F(t) for t in item['row_probabilities_exact']]
                y = [F(t) for t in item['col_probabilities_exact']]
                assert is_nash(a.tolist(), b.tolist(), x, y)


def test_partial_budget_never_promoted_to_complete():
    a = np.zeros((4, 4))
    b = np.zeros((4, 4))
    for fast_path in (True, False):
        result = search_exact_nash_cells(
            a, b, max_pairs=12, max_support=2, max_operations=200000,
            prune_supports=False, feasibility_fast_path=fast_path)
        assert result['status'] == 'partial'
        assert result['incomplete_reason'] == 'pair_budget'
        assert not result['complete_equilibrium_set']
