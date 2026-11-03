import pytest
from fastapi import HTTPException
from cfr_plus import CFRPlusRequest, solve_cfr_plus
from benchmark_cfr_phase34 import matching_pennies, run


def test_cross_player_infoset_collision_rejected():
    nodes = matching_pennies()
    nodes['h']['infoset'] = 'row'
    nodes['t']['infoset'] = 'row'
    with pytest.raises(HTTPException) as error:
        solve_cfr_plus(CFRPlusRequest(nodes=nodes, iterations=2))
    assert error.value.status_code == 422


def test_convergence_certificate():
    result = solve_cfr_plus(CFRPlusRequest(nodes=matching_pennies(), iterations=1000))
    assert result['exploitability'] < .1
    assert abs(sum(result['average_strategies']['row'].values())-1) < 1e-12


def test_benchmark_harness():
    result = run(iterations=(10, 20), repeats=2)
    assert len(result['results']) == 2
    assert all(item['median_ms'] >= 0 for item in result['results'])
