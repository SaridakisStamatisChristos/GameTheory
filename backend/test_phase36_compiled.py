import pytest
from fastapi import HTTPException
from cfr_plus import CFRPlusRequest, solve_cfr_plus
from cfr_plus_compiled import solve_cfr_plus_compiled
from benchmark_kuhn_phase35 import kuhn_poker

@pytest.mark.parametrize('iterations', [1, 10, 100, 500])
def test_compiled_matches_reference_kuhn(iterations):
    req = CFRPlusRequest(nodes=kuhn_poker(), iterations=iterations)
    a, b = solve_cfr_plus(req), solve_cfr_plus_compiled(req)
    assert abs(a['exploitability'] - b['exploitability']) < 1e-10
    assert abs(a['row_expected_payoff'] - b['row_expected_payoff']) < 1e-10
    for name in a['average_strategies']:
        for action, prob in a['average_strategies'][name].items():
            assert abs(prob - b['average_strategies'][name][action]) < 1e-10

def test_compiled_rejects_invalid_game():
    req = CFRPlusRequest(nodes={'root': {'player': 0, 'infoset': 'x', 'actions': {'a': 'missing'}}})
    with pytest.raises(HTTPException):
        solve_cfr_plus_compiled(req)
