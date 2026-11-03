"""Linear-time-in-tree best response for perfect-recall extensive-form games.

Counterfactual reach weights exclude the responding player's actions.  At an
information set, the same action must be selected at every member node.
"""
from collections import defaultdict


def best_response(nodes, root, profile, player):
    members = defaultdict(list)
    counterfactual = {}
    children = {}

    def collect(k, reach):
        n = nodes[k]
        counterfactual[k] = reach
        if 'payoff' in n:
            return
        owner = n['player']
        children[k] = list(n['actions'].values())
        if owner == player:
            members[n['infoset']].append(k)
        for a, child in n['actions'].items():
            weight = (n['probabilities'][a] if owner == 'chance'
                      else profile[n['infoset']][a] if owner != player else 1.)
            collect(child, reach * weight)

    collect(root, 1.)
    decisions = {}
    # A responding player's descendant information sets must be resolved first.
    # Perfect recall guarantees that all nodes in one infoset share own history.
    pending = set(members)

    def value(k):
        n = nodes[k]
        if 'payoff' in n:
            return float(n['payoff'])
        owner = n['player']
        if owner == player:
            return value(n['actions'][decisions[n['infoset']]])
        probs = n['probabilities'] if owner == 'chance' else profile[n['infoset']]
        return sum(probs[a] * value(child) for a, child in n['actions'].items())

    def unresolved_descendant(k, own):
        n = nodes[k]
        if 'payoff' in n:
            return False
        if n['player'] == player and n['infoset'] != own and n['infoset'] in pending:
            return True
        return any(unresolved_descendant(c, own) for c in n['actions'].values())

    while pending:
        ready = [i for i in sorted(pending) if not any(
            unresolved_descendant(c, i) for k in members[i]
            for c in nodes[k]['actions'].values())]
        if not ready:
            raise ValueError('Unable to order information sets')
        for i in ready:
            actions = list(nodes[members[i][0]]['actions'])
            scores = {}
            for a in actions:
                scores[a] = sum(counterfactual[k] * value(nodes[k]['actions'][a])
                                for k in members[i])
            decisions[i] = max(actions, key=lambda a: scores[a] if player == 0 else -scores[a])
            pending.remove(i)
    return value(root), decisions
