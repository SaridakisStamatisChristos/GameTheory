"""v55 exact support fast-path vs baseline research contracts."""
import pytest
from benchmark_exact_scaling_v55 import fixtures, signature, run


def test_fixture_determinism_and_adversarial_distributions():
    a=fixtures()
    b=fixtures()
    assert len(a)==15
    assert [n for n,_,_ in a]==[n for n,_,_ in b]
    assert all((x==xx).all() and (y==yy).all() for (_,x,y),(_,xx,yy) in zip(a,b))
    assert any(n.startswith('all_ties_') for n,_,_ in a)
    assert any(n.startswith('dominated_') for n,_,_ in a)


def test_signature_distinguishes_witnesses():
    a={'families':[{'support':{'row':[0],'col':[1]},
                    'row_probabilities_exact':['1','0'],
                    'col_probabilities_exact':['0','1']}]}
    b={'families':[]}
    assert signature(a)!=signature(b)


@pytest.mark.parametrize('bad',[0,11,True,1.2])
def test_bad_repetition_rejected(bad):
    with pytest.raises(ValueError):
        run(repetitions=bad)


def test_paired_campaign_preserves_all_supports():
    result=run(repetitions=1)
    assert len(result['cases'])==15
    assert all(c['support_sets_identical'] for c in result['cases'])
    assert any(c['uniform_hits']>0 for c in result['cases'])
    assert any(not c['complete'] for c in result['cases'])
    assert all(c['baseline_lp_calls']>=c['fast_lp_calls'] for c in result['cases'])
