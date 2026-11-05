"""v56 runnable measurement contract without optional research dependencies."""
import pytest
from benchmark_sign_screen_v56 import fixtures, run


def test_seeded_sign_fixture_protocol():
    original = fixtures(56056)
    again = fixtures(56056)
    assert len(original) == 15
    assert [s for s, _, _ in original] == [s for s, _, _ in again]
    assert all((a == a2).all() and (b == b2).all()
               for (_,a,b),(_,a2,b2) in zip(original, again))


@pytest.mark.parametrize('invalid',[0,11,1.25,True])
def test_campaign_validation_fails_closed(invalid):
    with pytest.raises(ValueError, match='repetitions'):
        run(repetitions=invalid)


def test_matched_performance_campaign_preserves_exact_support_semantics():
    out = run(repetitions=1)
    assert len(out['cases']) == 15
    assert len(out['fixture_sha256']) == 64
    assert all(x['exact_support_sets_equal'] for x in out['cases'])
    assert all(x['baseline_simplex_lp_calls'] >= x['screen_simplex_lp_calls']
               for x in out['cases'])
    assert all(x['screen_tracemalloc_peak_bytes'] >= 0
               and x['baseline_tracemalloc_peak_bytes'] >= 0 for x in out['cases'])
    assert any(x['screen_rejections'] > 0 for x in out['cases'])
    assert any(not x['complete'] for x in out['cases'])
