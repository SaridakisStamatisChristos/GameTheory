"""v58 multiplayer exact benchmark contract gates."""
import pytest
from benchmark_multiplayer_v58 import fixtures, run


def test_benchmark_fixture_determinism():
    first=fixtures()
    second=fixtures()
    assert len(first)==12
    assert [n for n,_ in first]==[n for n,_ in second]
    assert all((a==b).all() for (_,a),(_,b) in zip(first,second))
    assert any(a.shape[0]==5 for _,a in first)


@pytest.mark.parametrize('invalid',[0,-1,11,True,1.5])
def test_bad_campaign_repetitions_are_rejected(invalid):
    with pytest.raises(ValueError,match='Repetitions'):
        run(repetitions=invalid)


def test_independent_highs_campaign_and_pure_reference_sets():
    out=run(repetitions=1)
    assert len(out['cases'])==12
    assert len(out['fixtures_sha256'])==64
    assert all(c['independent_pure_sets_equal'] for c in out['cases'])
    assert all(c['independent_highs_abs_error']<1e-8 for c in out['cases'])
    assert all(c['exact_operations']>0 for c in out['cases'])
