from benchmark_kuhn_phase35 import kuhn_poker, run
from cfr_plus import CFRPlusRequest, solve_cfr_plus


def test_kuhn_poker_known_value_and_exploitability():
    result = solve_cfr_plus(CFRPlusRequest(nodes=kuhn_poker(), iterations=3000))
    assert abs(result['row_expected_payoff'] + 1 / 18) < 0.035
    assert result['exploitability'] < 0.06
    assert len(result['average_strategies']) == 12
    for strategy in result['average_strategies'].values():
        assert abs(sum(strategy.values()) - 1) < 1e-10


def test_kuhn_benchmark_schema():
    output = run(iterations=(20,), repeats=2)
    assert output['nodes'] > 30
    assert output['results'][0]['exploitability'] >= 0
