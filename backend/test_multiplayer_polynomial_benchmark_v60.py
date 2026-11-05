"""v60 reproducible polynomial enumeration benchmark and evidence semantics."""
import pytest
from benchmark_multiplayer_polynomial_v60 import fixtures,run


def test_fixture_determinism_and_declared_degenerate_cases():
    first=fixtures()
    second=fixtures()
    assert len(first)==18
    assert [name for name,*_ in first]==[name for name,*_ in second]
    assert all((x[1]==y[1]).all() for x,y in zip(first,second))
    assert all(first[k][3] is False for k in (7,8,9))


@pytest.mark.parametrize('invalid',[-1,0,True,1.5,11])
def test_bad_repetitions_rejected(invalid):
    with pytest.raises(ValueError,match='repetitions'):
        run(repetitions=invalid)


def test_full_eighteen_game_exact_and_numerical_campaign():
    result=run(repetitions=1)
    assert len(result['cases'])==18
    assert len(result['fixtures_sha256'])==64
    assert all(z['support_cells_processed']==27 for z in result['cases'])
    assert result['cases'][0]['fully_mixed_algebraic_roots']==2
    assert result['cases'][0]['exact_completeness_claim']
    assert result['cases'][7]['unresolved_support_cells']>=1
    assert all(z['maximum_independent_numerical_unilateral_regret']<=1e-8
               for z in result['cases'])
