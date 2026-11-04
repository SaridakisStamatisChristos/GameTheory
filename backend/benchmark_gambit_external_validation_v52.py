"""Required actual Gambit run, collected explicitly by the dedicated CI job only."""
from benchmark_gambit_v52 import run, fixtures, fixture_digest


def test_real_gambit_reference_all_fixtures():
    result = run(repetitions=2)
    assert result['pygambit_version'] == '17.0.0b2'
    assert result['evidence'] == 'executed_external_engine'
    assert result['fixtures_sha256'] == fixture_digest(fixtures())
    assert len(result['cases']) == len(fixtures())
    assert all(c['both_exact_certificates_passed'] for c in result['cases'])
    assert all(len(c['gambit_all_seconds']) == 2 for c in result['cases'])


