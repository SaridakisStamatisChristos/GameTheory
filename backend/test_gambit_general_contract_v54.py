"""Offline independent exact Nash membership / degeneracy correctness gates."""
from fractions import Fraction
import numpy as np
import pytest
from benchmark_gambit_general_v54 import (
    fixtures, fixture_digest, nfg_bytes, independent_exact_nash,
    in_h_polytope, containing_families, independently_pure_supports,
    compare_case, run,
)


def test_deterministic_fixture_campaign():
    cases = fixtures()
    assert len(cases) >= 18
    assert fixture_digest(cases) == fixture_digest(fixtures())
    assert len(set(name for name, _, _ in cases)) == len(cases)
    assert any(a.shape != (2,2) for _, a, _ in cases)
    assert all(a.shape == b.shape and a.size <= 16 for _,a,b in cases)


def test_nfg_payoff_order_is_column_major_and_exact():
    a = np.array([[1, 2], [3, 4]], dtype=int)
    b = np.array([[-2, -3], [4, 5]], dtype=int)
    txt = nfg_bytes(a,b).decode()
    assert txt.startswith('NFG 1 R "GameTheory v54"')
    assert txt.split('}',2)[-1].strip() == '1 -2 3 4 2 -3 4 5'
    with pytest.raises(ValueError):
        nfg_bytes(np.array([[0.5]]), np.array([[0]]))
    with pytest.raises(ValueError):
        nfg_bytes(a, b[:, :1])


def test_original_payoff_auditor_detects_profitable_deviations():
    a = [[2, 0], [0, 1]]
    b = [[1, 0], [0, 2]]
    assert independent_exact_nash(a,b,[1,0],[1,0]) == (2,1)
    assert independent_exact_nash(a,b,[0,1],[0,1]) == (1,2)
    assert independent_exact_nash(a,b,
                  [Fraction(2,3),Fraction(1,3)],
                  [Fraction(1,3),Fraction(2,3)]) == (Fraction(2,3),Fraction(2,3))
    for x,y in (([1,0],[0,1]), ([Fraction(1,2)]*2, [0,1])):
        with pytest.raises(AssertionError,match='unilateral-deviation'):
            independent_exact_nash(a,b,x,y)
    with pytest.raises(AssertionError,match='simplex'):
        independent_exact_nash(a,b,[Fraction(1,2)]*2,[1,1])
    with pytest.raises(AssertionError,match='dimensions'):
        independent_exact_nash(a,b,[1],[1])


def test_h_polytope_strict_support_and_inequality_membership():
    desc = {'variable_actions':[0,1],
            'equalities_dot_coefficients_eq_zero':[
                {'coefficients_exact':['1','-1']}],
            'inequalities_dot_coefficients_ge_zero':[
                {'coefficients_exact':['1','0']}],
            }
    half = [Fraction(1,2),Fraction(1,2),Fraction(0)]
    assert in_h_polytope(desc,half)
    assert not in_h_polytope(desc,[Fraction(1),Fraction(0),Fraction(0)])
    assert not in_h_polytope(desc,[Fraction(1,2),Fraction(1,4),Fraction(1,4)])
    assert not in_h_polytope(desc,[Fraction(1,4),Fraction(3,4),Fraction(0)])
    dec = {'variable_actions':[0],
           'equalities_dot_coefficients_eq_zero':[],
           'inequalities_dot_coefficients_ge_zero':[
               {'coefficients_exact':['-1']}],
           }
    assert not in_h_polytope(dec,[Fraction(1),Fraction(0)])
    family = {'row_strategy_polytope':desc,'col_strategy_polytope':desc}
    assert len(containing_families([family],half,half)) == 1
    assert containing_families([family],half,[Fraction(1),0,0]) == []


def test_direct_pure_scan_honors_general_sum():
    a=np.asarray([[2,0],[0,1]])
    b=np.asarray([[1,0],[0,2]])
    assert independently_pure_supports(a,b)=={(0,0),(1,1)}


def test_invalid_repetition_fails_before_importing_gambit():
    for n in (0, -1, 21, 1.5, True):
        with pytest.raises(ValueError,match='repetitions'):
            run(n)
        with pytest.raises(ValueError,match='repetitions'):
            compare_case(None,'invalid',np.asarray([[1]]),np.asarray([[1]]),n)
