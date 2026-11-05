"""Adversarial fail-closed tests for rational simplex internal invariants."""
from fractions import Fraction as F
import pytest
from exact_rational_lp import _RationalSimplex, feasible_strict_strategy


def test_standard_form_unbounded_is_never_misreported_as_feasible_support():
    # The general internal simplex detects unboundedness; support LPs always
    # bound their t objective by 1, so this is a defense-in-depth test.
    with pytest.raises(AssertionError,match='unbounded'):
        _RationalSimplex([],[],[F(1)],{'steps':0,'limit':1000}).solve()


def test_phase_one_artificial_variable_cleanup(monkeypatch):
    # Construct the auxiliary-basis cleanup state explicitly; it can arise
    # with redundant degenerate systems. Verify cleanup preserves a solution.
    original=_RationalSimplex._phase

    def force_artificial_basic(self,phase):
        if phase==1:
            self.D[self.m+1][self.n+1]=F(0)
            return True
        return original(self,phase)

    monkeypatch.setattr(_RationalSimplex,'_phase',force_artificial_basic)
    candidate=_RationalSimplex([[F(1)],[-F(1)]],[F(1),-F(1)],
                               [F(1)],{'steps':0,'limit':10000}).solve()
    assert candidate==[F(1)]


def test_bad_probability_is_rejected_before_any_best_response_claim(monkeypatch):
    monkeypatch.setattr(_RationalSimplex,'solve',lambda self:[F(0),F(1)])
    with pytest.raises(AssertionError,match='Invalid rational'):
        feasible_strict_strategy([[F(0)]],(0,),(0,),
                                 {'steps':0,'limit':1000}, fast_path=False)


def test_false_best_response_is_rejected_by_independent_verifier(monkeypatch):
    monkeypatch.setattr(_RationalSimplex,'solve',
                        lambda self:[F(1,2),F(1,2),F(1,2)])
    with pytest.raises(AssertionError,match='best-response'):
        feasible_strict_strategy([[F(0),F(0)],[F(1),F(1)]],
                                 (0,),(0,1),{'steps':0,'limit':1000},
                                 sign_screen=False)


def test_odd_infeasible_negative_rhs_lp():
    solver=_RationalSimplex([[F(0)]],[F(-1)],[F(1)],
                            {'steps':0,'limit':10000})
    assert solver.solve() is None
