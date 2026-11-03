"""Pre-indexed pure-Python CFR+ traversal kernel (not native compiled code).

Shares the validated game contract with cfr_plus; returns the same certificate.
"""
from cfr_plus import CFRPlusRequest, validate_cfr_game
from best_response import best_response


def solve_cfr_plus_compiled(req: CFRPlusRequest):
    # Validate once without executing an unnecessary CFR+ training iteration.
    signatures = validate_cfr_game(req)
    nodes = req.nodes
    infos = list(signatures)
    info_index = {name: i for i, name in enumerate(infos)}
    actions = [list(action_names) for _, action_names in signatures.values()]
    order = []
    def compile_node(k):
        n = nodes[k]
        index = len(order)
        order.append(None)
        if 'payoff' in n:
            order[index] = (-1, -1, (), (), float(n['payoff']))
        else:
            p = n['player']
            labels = list(n['actions'])
            children = tuple(compile_node(n['actions'][a]) for a in labels)
            if p == 'chance':
                order[index] = (2, -1, children, tuple(n['probabilities'][a] for a in labels), 0.)
            else:
                ii = info_index[n['infoset']]
                # Actions in a given infoset are sorted consistently by validation.
                positions = tuple(actions[ii].index(a) for a in labels)
                order[index] = (p, ii, children, positions, 0.)
        return index
    root = compile_node(req.root)
    regrets = [[0.] * len(a) for a in actions]
    sums = [[0.] * len(a) for a in actions]
    def traverse(idx, r0, r1, chance, player, policy, delta, weights, t):
        p, ii, children, extra, payoff = order[idx]
        if p == -1:
            return payoff
        if p == 2:
            return sum(prob * traverse(c, r0, r1, chance * prob, player, policy, delta, weights, t)
                       for c, prob in zip(children, extra))
        strat = policy[ii]
        vals = [traverse(c, r0 * (strat[a] if p == 0 else 1.),
                         r1 * (strat[a] if p == 1 else 1.), chance, player, policy, delta, weights, t)
                for c, a in zip(children, extra)]
        value = sum(strat[a] * v for a, v in zip(extra, vals))
        if p == player:
            cf = chance * (r1 if p == 0 else r0)
            own = r0 if p == 0 else r1
            sign = 1. if p == 0 else -1.
            for a, v in zip(extra, vals):
                delta[ii][a] += cf * (v - value) * sign
                weights[ii][a] += t * chance * own * strat[a]
        return value
    for t in range(1, req.iterations + 1):
        for player in (0, 1):
            policy = []
            for row in regrets:
                total = sum(row)
                policy.append([v / total for v in row] if total > 0 else [1. / len(row)] * len(row))
            delta = [[0.] * len(a) for a in actions]
            weights = [[0.] * len(a) for a in actions]
            traverse(root, 1., 1., 1., player, policy, delta, weights, t)
            for i, row in enumerate(regrets):
                for j in range(len(row)):
                    row[j] = max(0., row[j] + delta[i][j])
                    sums[i][j] += weights[i][j]
    avg = {}
    for i, name in enumerate(infos):
        total = sum(sums[i])
        probs = [v / total for v in sums[i]] if total else [1. / len(actions[i])] * len(actions[i])
        avg[name] = dict(zip(actions[i], probs))
    def evaluate(idx):
        p, ii, children, extra, payoff = order[idx]
        if p == -1:
            return payoff
        if p == 2:
            return sum(prob * evaluate(c) for c, prob in zip(children, extra))
        return sum(avg[infos[ii]][actions[ii][a]] * evaluate(c) for c, a in zip(children, extra))
    value = evaluate(root)
    br0, _ = best_response(nodes, req.root, avg, 0)
    br1, _ = best_response(nodes, req.root, avg, 1)
    return {'method': 'alternating CFR+ with linear averaging (pre-indexed Python tree)',
            'iterations': req.iterations, 'average_strategies': avg,
            'row_expected_payoff': value, 'best_response_row_payoff': br0,
            'best_response_col_payoff_to_row': br1,
            'exploitability': max(0., br0 - br1),
            'certificate_status': 'exact_perfect_recall_tree_best_response'}
