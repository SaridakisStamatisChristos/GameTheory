"""Budgeted, dimension-independent exact rational Nash-set support search.

Searches supports in increasing cardinality, builds exact-rational
best-response H-cells, and never labels a truncated search as complete.
Uses the existing rational vertex feasibility engine without a size ceiling.
"""
from fractions import Fraction
from itertools import combinations
import numpy as np
from exact_support import _feasible_strategy_by_vertices, _verify_exact_equilibrium, ExactWorkLimit
from exact_nash_set import _player_polytope


def search_exact_nash_cells(a, b, *, max_pairs=96, max_operations=20000,
                            max_support=3):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (a.ndim != 2 or a.size == 0 or b.shape != a.shape or
            not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Expected matching nonempty finite matrices')
    if (type(max_pairs) is not int or not 1 <= max_pairs <= 4096 or
            type(max_operations) is not int or not 1 <= max_operations <= 2000000 or
            type(max_support) is not int or not 1 <= max_support <= 20):
        raise ValueError('Invalid exact-search work budget')
    m, n = a.shape
    total = (2**m-1)*(2**n-1)
    af = [[Fraction(str(t)) for t in row] for row in a]
    bf = [[Fraction(str(t)) for t in row] for row in b]
    bt = [list(row) for row in zip(*bf)]
    work = {'steps': 0, 'limit': max_operations}
    families = []
    checked = 0
    stopped = None
    # Small cardinality first, maximizing useful discoveries on large games.
    for ki in range(1, min(m, max_support)+1):
        if stopped:
            break
        for kj in range(1, min(n, max_support)+1):
            if stopped:
                break
            for I in combinations(range(m), ki):
                if stopped:
                    break
                for J in combinations(range(n), kj):
                    if checked == max_pairs:
                        stopped = 'pair_budget'
                        break
                    try:
                        y = _feasible_strategy_by_vertices(af, I, J, work)
                        x = _feasible_strategy_by_vertices(bt, J, I, work) if y is not None else None
                    except ExactWorkLimit:
                        stopped = 'operation_budget'
                        break
                    checked += 1
                    if x is None:
                        continue
                    xx = [Fraction(0)]*m
                    yy = [Fraction(0)]*n
                    for i,p in zip(I,x):
                        xx[i] = p
                    for j,p in zip(J,y):
                        yy[j] = p
                    if not _verify_exact_equilibrium(af,bf,xx,yy):
                        raise AssertionError('Exact-search witness failed Nash verification')
                    families.append({
                        'support': {'row': list(I), 'col': list(J)},
                        'row_probabilities_exact': [str(t) for t in xx],
                        'col_probabilities_exact': [str(t) for t in yy],
                        'row_strategy_polytope': _player_polytope(bt,J,I,[str(t) for t in x]),
                        'col_strategy_polytope': _player_polytope(af,I,J,[str(t) for t in y]),
                    })
    complete = stopped is None and checked == total
    return {
        'support_pairs_total': total,
        'support_pairs_checked': checked,
        'exact_operations': work['steps'],
        'feasible_cells_found': len(families),
        'families': families,
        'complete_equilibrium_set': complete,
        'status': 'complete' if complete else 'partial',
        'incomplete_reason': None if complete else (stopped or 'support_size_cap'),
        'representation': 'exact rational strict-support product H-polytopes',
        'input_model': 'decimal strings of validated float64 inputs',
        'note': 'Unexplored supports are not classified. Only a completed full search proves set completeness.',
    }
