"""Offline comparator contracts; no optional OpenSpiel dependency in core CI."""
from fractions import Fraction
import numpy as np
import pytest
from benchmark_gambit_v52 import fixtures as gambit_fixtures, fixture_digest
from benchmark_openspiel_v53 import fixtures, model_translation, independent_numeric_saddle, run, compare_case


def test_fixture_composition_and_determinism():
    games = fixtures()
    assert len(games) == len(gambit_fixtures()) + 3
    assert fixture_digest(games) == fixture_digest(fixtures())
    assert any(a.shape == (10, 10) for _, a in games)
    assert len({name for name, _ in games}) == len(games)
    assert all(np.issubdtype(a.dtype, np.integer) and a.size <= 100 for _, a in games)


def test_translation_exact_on_large_offset_and_negative_payoffs():
    a = np.asarray([[100000, 99999], [99999, 100000]])
    transformed, offset = model_translation(a)
    assert offset == 99999
    np.testing.assert_array_equal(transformed, [[1, 0], [0, 1]])
    negative = np.asarray([[-2, 3], [-1, -4]])
    shifted, minus_four = model_translation(negative)
    assert minus_four == -4
    np.testing.assert_array_equal(shifted, negative + 4)


def test_numeric_auditor_accepts_valid_profile_and_rejects_invalid():
    a = [[1, -1], [-1, 1]]
    value = Fraction(0)
    outcome = independent_numeric_saddle(a, [.5, .5], [.5, .5], value, 0., 0.)
    assert outcome['unilateral_gap_original_units'] == 0
    assert outcome['max_value_error_original_units'] == 0
    wrong = (
        ([1, 0], [1, 0], 0., 0.),
        ([0.3, 0.3], [.5, .5], 0., 0.),
        ([.5, .5], [.5, .5], 2., 0.),
        ([-1., 2.], [.5, .5], 0., 0.),
    )
    for x, y, rv, cv in wrong:
        with pytest.raises(AssertionError):
            independent_numeric_saddle(a, x, y, value, rv, cv)
    with pytest.raises(AssertionError, match='Invalid numerical'):
        independent_numeric_saddle(a, [float('nan'), 0.], [.5, .5], value, 0, 0)


def test_validation_and_fail_closed_numerical_values():
    for payoff in (np.zeros((0, 2), dtype=int), np.zeros((11, 11), dtype=int),
                   np.ones((2, 2), dtype=float), np.asarray([[1000001]], dtype=int)):
        with pytest.raises(ValueError):
            model_translation(payoff)
    with pytest.raises(ValueError, match='Translated'):
        model_translation(np.asarray([[-1000000, 1000000]], dtype=int))
    for number in (0, 31, -1, True, 0.5):
        with pytest.raises(ValueError, match='repetitions'):
            run(repetitions=number)
        with pytest.raises(ValueError, match='repetitions'):
            compare_case(None, None, 'bad', np.asarray([[1]], dtype=int), number)
