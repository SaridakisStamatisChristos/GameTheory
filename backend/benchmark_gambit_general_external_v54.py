"""Actual pinned Gambit rational extreme-equilibrium acceptance test (separate CI)."""
from benchmark_gambit_general_v54 import run, fixtures, fixture_digest


def test_real_gambit_exact_general_sum_all_fixtures():
    out = run(repetitions=2)
    assert out['pygambit'] == '17.0.0b2'
    assert out['fixture_sha256'] == fixture_digest(fixtures())
    assert len(out['cases']) == len(fixtures())
    assert all(item['all_reference_points_exactly_certified_and_contained']
               for item in out['cases'])
    assert all(item['sampling_repetitions'] == 2 for item in out['cases'])
