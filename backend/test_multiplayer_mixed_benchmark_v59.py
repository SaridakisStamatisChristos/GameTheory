"""v59 benchmark completeness and independent rational Nash checks."""
import pytest
from benchmark_multiplayer_mixed_v59 import fixtures,run


def test_seeded_multiplayer_fixture_stability():
    x=fixtures()
    y=fixtures()
    assert len(x)==10
    assert [name for name,_ in x]==[name for name,_ in y]
    assert all((a==b).all() for (_,a),(_,b) in zip(x,y))
    assert any(a.shape[0]==5 for _,a in x)


@pytest.mark.parametrize('bad',[-1,0,True,1.5,11])
def test_invalid_benchmark_repetitions(bad):
    with pytest.raises(ValueError,match='repetitions'):
        run(repetitions=bad)


def test_verified_rational_multiplayer_mixed_campaign():
    out=run(repetitions=1)
    assert len(out['cases'])==10
    assert len(out['fixtures_sha256'])==64
    assert all(c['independent_rational_deviation_verified'] for c in out['cases'])
    assert all(c['independent_pure_nash_sets_equal'] for c in out['cases'])
    assert out['cases'][0]['mixed_candidates_found']>0
    assert out['cases'][1]['mixed_candidates_found']>0
