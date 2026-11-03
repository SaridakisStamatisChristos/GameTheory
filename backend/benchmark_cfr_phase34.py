"""Deterministic CFR+ convergence benchmark with full certificate checks."""
import argparse
import json
import statistics
import time
from cfr_plus import CFRPlusRequest, solve_cfr_plus


def matching_pennies():
    nodes = {'root': {'player': 0, 'infoset': 'row', 'actions': {'H': 'h', 'T': 't'}}}
    for row, node in [('H', 'h'), ('T', 't')]:
        nodes[node] = {'player': 1, 'infoset': 'col', 'actions': {'H': node+'h', 'T': node+'t'}}
        for col in 'HT':
            nodes[node+col.lower()] = {'payoff': 1 if row == col else -1}
    return nodes


def run(iterations=(50, 100, 250, 500, 1000), repeats=3):
    results = []
    for count in iterations:
        latencies, gaps = [], []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = solve_cfr_plus(CFRPlusRequest(nodes=matching_pennies(), iterations=count))
            latencies.append((time.perf_counter_ns()-start)/1e6)
            gaps.append(result['exploitability'])
            assert result['certificate_status'] == 'exact_perfect_recall_tree_best_response'
            assert 0 <= gaps[-1] <= 2.000001
        results.append({'iterations': count, 'median_ms': statistics.median(latencies),
                        'min_ms': min(latencies), 'max_ms': max(latencies),
                        'exploitability': statistics.median(gaps), 'repetitions': repeats})
    return {'game': 'matching_pennies', 'method': 'alternating_cfr_plus', 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', default='benchmark_cfr_phase34_results.json')
    args = parser.parse_args()
    data = run(repeats=args.repeats)
    with open(args.output, 'w') as file:
        json.dump(data, file, indent=2)
    print(json.dumps(data, indent=2))
