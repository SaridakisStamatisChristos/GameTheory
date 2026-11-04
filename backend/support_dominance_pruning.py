"""Sound strict-support Nash pruning by conditional pure weak dominance.

For any support (I,J), y_j > 0 on every j in J.  If an alternative row k
weakly dominates a supported row i over J and is strictly better at at least
one j in J, k has strictly higher expected payoff for every admissible y.
Thus (I,J) cannot contain a Nash equilibrium.  The column rule is symmetric.

Precompute integer masks for each pure-action comparison. The masks encode
only the sign of validated float64 payoffs, which agrees with their decimal
rational interpretation. The rules never use approximate thresholds.
"""
import numpy as np


class ConditionalDominancePruner:
    """Conservative exact-sign necessary-condition filter for support pairs."""

    def __init__(self, a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.ndim != 2 or a.size == 0 or b.shape != a.shape or not (
                np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError('Expected finite nonempty matching matrices')
        m, n = a.shape
        self.row_witnesses = [[] for _ in range(m)]
        self.col_witnesses = [[] for _ in range(n)]
        for i in range(m):
            for k in range(m):
                if i == k:
                    continue
                ge = sum((1 << j) for j in range(n) if a[k, j] >= a[i, j])
                gt = sum((1 << j) for j in range(n) if a[k, j] > a[i, j])
                if gt:
                    self.row_witnesses[i].append((k, ge, gt))
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                ge = sum((1 << i) for i in range(m) if b[i, k] >= b[i, j])
                gt = sum((1 << i) for i in range(m) if b[i, k] > b[i, j])
                if gt:
                    self.col_witnesses[j].append((k, ge, gt))
        self._row_cache = {}
        self._col_cache = {}

    def _excluded_rows(self, column_mask):
        if column_mask not in self._row_cache:
            certificates = {}
            for i, alternatives in enumerate(self.row_witnesses):
                for k, weak_mask, strict_mask in alternatives:
                    if column_mask & ~weak_mask == 0 and column_mask & strict_mask:
                        certificates[i] = k
                        break
            self._row_cache[column_mask] = certificates
            return certificates
        return self._row_cache[column_mask]

    def _excluded_columns(self, row_mask):
        if row_mask not in self._col_cache:
            certificates = {}
            for j, alternatives in enumerate(self.col_witnesses):
                for k, weak_mask, strict_mask in alternatives:
                    if row_mask & ~weak_mask == 0 and row_mask & strict_mask:
                        certificates[j] = k
                        break
            self._col_cache[row_mask] = certificates
            return certificates
        return self._col_cache[row_mask]

    def explain(self, row_support, column_support):
        """Return a mathematical exclusion witness, or None if inconclusive.

        A None result is *not* evidence that a support is feasible.
        Supports must be nonempty; callers enumerate only nonempty supports.
        """
        row_mask = sum(1 << i for i in row_support)
        col_mask = sum(1 << j for j in column_support)
        rows = self._excluded_rows(col_mask)
        for i in row_support:
            if i in rows:
                return {'player': 'row', 'dominated_action': i,
                        'dominates_action': rows[i]}
        cols = self._excluded_columns(row_mask)
        for j in column_support:
            if j in cols:
                return {'player': 'col', 'dominated_action': j,
                        'dominates_action': cols[j]}
        return None
