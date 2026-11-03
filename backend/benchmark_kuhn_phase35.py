"""Kuhn poker: nontrivial imperfect-information CFR+ regression and timing.

Each player antes one chip. Cards J<Q<K, no replacement. Bet size one.
Payoffs are net chips to player 0; the equilibrium game value is -1/18.
"""
import json
import statistics
import time
from cfr_plus import CFRPlusRequest, solve_cfr_plus


def kuhn_poker():
    cards = 'JQK'
    nodes = {'root': {'player': 'chance', 'actions': {}, 'probabilities': {}}}
    for c0 in cards:
        for c1 in cards:
            if c0 == c1:
                continue
            deal = c0 + c1
            nodes['root']['actions'][deal] = 'deal_' + deal
            nodes['root']['probabilities'][deal] = 1 / 6
            def terminal(history, payoff):
                key = deal + '_' + history
                nodes[key] = {'payoff': payoff}
                return key
            def decision(history, player, actions):
                key = deal + '_' + (history or 'start')
                nodes[key] = {'player': player, 'infoset': (c0 if player == 0 else c1) + ':' + history,
                              'actions': actions}
                return key
            winner = 1 if cards.index(c0) > cards.index(c1) else -1
            cc = terminal('cc', winner)
            bc = terminal('bc', winner * 2)
            bf = terminal('bf', 1)
            cbc = terminal('cbc', winner * 2)
            cbf = terminal('cbf', -1)
            decision('cb', 0, {'call': cbc, 'fold': cbf})
            decision('c', 1, {'check': cc, 'bet': deal + '_cb'})
            decision('b', 1, {'call': bc, 'fold': bf})
            decision('', 0, {'check': deal + '_c', 'bet': deal + '_b'})
            nodes['root']['actions'][deal] = deal + '_start'
    return nodes


def run(iterations=(100, 500, 2000), repeats=3):
    nodes = kuhn_poker()
    results = []
    for count in iterations:
        times, gaps, values = [], [], []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = solve_cfr_plus(CFRPlusRequest(nodes=nodes, iterations=count))
            times.append((time.perf_counter_ns() - start) / 1e6)
            gaps.append(result['exploitability'])
            values.append(result['row_expected_payoff'])
        results.append({'iterations': count, 'repeats': repeats, 'median_ms': statistics.median(times),
                        'p90_ms': sorted(times)[min(len(times)-1, int(.9*len(times)))],
                        'exploitability': statistics.median(gaps),
                        'value': statistics.median(values), 'value_error': abs(statistics.median(values) + 1/18)})
    return {'game': 'kuhn_poker', 'nodes': len(nodes), 'results': results}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
