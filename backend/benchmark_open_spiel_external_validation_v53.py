"""Mandatory OpenSpiel external LP gate, kept out of core coverage denominator."""
from benchmark_openspiel_v53 import run, fixtures, fixture_digest


def test_real_open_spiel_comparison_complete():
    result = run(repetitions=2)
    assert result['open_spiel_version'] == '2.0.2'
    assert result['result'] == 'all_numerical_candidates_audited'
    assert len(result['cases']) == len(fixtures())
    assert result['fixture_digest_sha256'] == fixture_digest(fixtures())
    assert all(c['open_spiel_certificate']['unilateral_gap_original_units'] <= 2e-5
               for c in result['cases'])
