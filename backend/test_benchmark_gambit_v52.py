"""Standalone comparator validation; actual Gambit calls are required in its CI job."""
from fractions import Fraction
import numpy as np
import pytest
from benchmark_gambit_v52 import fixtures, fixture_digest, exact_certificate, nfg_bytes, run


def test_fixture_suite_stable_and_exact_input_contract():
    first = fixtures()
    assert fixture_digest(first) == fixture_digest(fixtures())
    assert len(first) >= 15
    assert len({name for name, _ in first}) == len(first)
    assert any(a.shape[0] != a.shape[1] for _, a in first)
    for _, a in first:
        assert np.issubdtype(a.dtype, np.integer)
        assert 1 <= a.shape[0] * a.shape[1] <= 100


def test_independent_certificate_catches_wrong_player_strategy():
    matching_pennies = np.array([[1, -1], [-1, 1]])
    assert exact_certificate(matching_pennies,
                             [Fraction(1, 2)]*2,
                             [Fraction(1, 2)]*2) == 0
    for x, y, message in (
        ([1, 0], [1, 0], 'saddle-point'),
        ([Fraction(1, 4), Fraction(1, 4)], [Fraction(1, 2)]*2, 'simplex'),
        ([-1, 2], [Fraction(1, 2)]*2, 'simplex'),
    ):
        with pytest.raises(AssertionError, match=message):
            exact_certificate(matching_pennies, x, y)
    with pytest.raises(ValueError, match='integer'):
        exact_certificate(np.array([[0.5]]), [1], [1])
    with pytest.raises(AssertionError, match='dimensions'):
        exact_certificate(matching_pennies, [1], [1])


def test_reject_invalid_benchmark_repetition_without_importing_gambit():
    for repetitions in (0, -1, 21, True, 2.5):
        with pytest.raises(ValueError, match='repetitions'):
            run(repetitions=repetitions)


def test_real_gambit_reference_all_fixtures():
    result = run(repetitions=2)
    assert result['pygambit_version'] == '17.0.0b2'
    assert result['evidence'] == 'executed_external_engine'
    assert result['fixtures_sha256'] == fixture_digest(fixtures())
    assert len(result['cases']) == len(fixtures())
    assert all(c['both_exact_certificates_passed'] for c in result['cases'])
    assert all(len(c['gambit_all_seconds']) == 2 for c in result['cases'])


def test_nfg_column_major_payoff_serialization():
    raw = nfg_bytes(np.asarray([[1, 2], [3, 4]], dtype=np.int64)).decode('ascii')
    assert raw.startswith('NFG 1 R "GameTheory v52"')
    assert raw.split('}', 2)[-1].strip() == '1 -1 3 -3 2 -2 4 -4'
    with pytest.raises(ValueError, match='integer matrix'):
        nfg_bytes(np.asarray([[0.5]]))
