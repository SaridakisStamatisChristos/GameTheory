"""Exact conditional mixed-strategy dominance certificates for Nash support pruning.

A supported action i cannot be a best response to any y with strictly
positive support J if a convex mixture over other own actions is weakly
better at every j in J and strictly better at some j in J.

Opt-in because discovering this witness solves an additional rational LP.
Successful certificates are checked with independent Fraction arithmetic.
"""
from fractions import Fraction
import numpy as np
from exact_rational_lp import _RationalSimplex


class ConditionalMixedDominancePruner:
    """Exact LP pruning with support-wise memoization and shared work budget."""

    def __init__(self, a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if (a.ndim != 2 or a.size == 0 or a.shape != b.shape or
                not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError('Expected finite nonempty matching matrices')
        self.payoffs = (
            [[Fraction(str(z)) for z in row] for row in a],
            [[Fraction(str(z)) for z in row] for row in b.T],
        )
        self.cache = {}
        self.lp_attempts = 0
        self.cache_hits = 0

    def _dominating_mixture(self, player, target, opponent_support, work):
        key = (player, target, tuple(opponent_support))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        matrix = self.payoffs[player]
        alternatives = tuple(i for i in range(len(matrix)) if i != target)
        if not alternatives:
            certificate = None
        elif any(max(matrix[i][j] for i in alternatives) < matrix[target][j]
                 for j in opponent_support):
            # A convex mixture cannot improve a coordinate beyond its max.
            certificate = None
        else:
            n = len(alternatives)
            constraints = [
                [Fraction(1)]*n,
                [Fraction(-1)]*n,
            ]
            rhs = [Fraction(1), Fraction(-1)]
            for j in opponent_support:
                constraints.append([-matrix[i][j] for i in alternatives])
                rhs.append(-matrix[target][j])
            objective = [sum(matrix[i][j] for j in opponent_support)
                         for i in alternatives]
            self.lp_attempts += 1
            weights = _RationalSimplex(constraints, rhs, objective, work).solve()
            certificate = None
            if weights is not None:
                if sum(weights) != 1 or any(w < 0 for w in weights):
                    raise AssertionError('Mixed-dominance LP returned invalid simplex weights')
                gains = [sum(w*matrix[i][j] for w,i in zip(weights, alternatives))
                         -matrix[target][j] for j in opponent_support]
                if any(g < 0 for g in gains):
                    raise AssertionError('Mixed-dominance LP violated payoff inequalities')
                if any(g > 0 for g in gains):
                    certificate = {
                        'player': 'row' if player == 0 else 'col',
                        'dominance_kind': 'mixed',
                        'dominated_action': target,
                        'dominating_mixture_exact': [
                            {'action': action, 'probability_exact': str(weight)}
                            for action,weight in zip(alternatives,weights) if weight > 0
                        ],
                        'strict_gain_actions': [int(j) for j,g in zip(opponent_support,gains) if g > 0],
                    }
        # Bounded cache avoids growth unrelated to the support-pair work budget.
        if len(self.cache) < 8192:
            self.cache[key] = certificate
        return certificate

    def explain(self, row_support, column_support, work):
        for i in row_support:
            cert = self._dominating_mixture(0, i, column_support, work)
            if cert is not None:
                return cert
        for j in column_support:
            cert = self._dominating_mixture(1, j, row_support, work)
            if cert is not None:
                return cert
        return None
