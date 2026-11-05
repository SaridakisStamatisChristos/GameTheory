"""v57 stochastic benchmarking and incomplete-result disclosure gate."""
import pytest
from benchmark_stochastic_v57 import fixtures,run


def test_fixed_and_seeded_fixture_determinism():
    original=fixtures()
    again=fixtures()
    assert len(original)==10
    assert [f[0] for f in original] == [f[0] for f in again]
    assert all((x[1]==y[1]).all() and (x[2]==y[2]).all()
               for x,y in zip(original,again))


@pytest.mark.parametrize('invalid',[0,11,True,1.5])
def test_invalid_benchmark_repetitions_fail_closed(invalid):
    with pytest.raises(ValueError,match='Repetitions'):
        run(repetitions=invalid)


def test_campaign_independent_reference_values_inside_bellman_bounds():
    out=run(repetitions=1)
    assert len(out['cases'])==10
    assert len(out['fixtures_sha256'])==64
    assert all(c['tolerance_met'] for c in out['cases'][:-1])
    assert out['cases'][-1]['status']=='partial_bound'
    assert all(c['reference_abs_error'] is None or
               c['reference_abs_error'] <= c['value_error_bound']+1e-8
               for c in out['cases'])
