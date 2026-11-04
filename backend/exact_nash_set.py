"""Exact finite union of product polytopes for the entire Nash set of small games.

For a feasible strict support pair (I,J), x and y are positive precisely on
their supports, normalized, and satisfy all rational best-response equalities
and inequalities. Every Nash pair has a unique such support pair. The union of
the reported Cartesian-product cells characterizes the full Nash set.
"""
from fractions import Fraction
import numpy as np
from exact_support import solve_exact_support
from exact_polytope_vertices import enumerate_polytope_vertices, VertexBudgetExceeded


def _player_polytope(payoffs, best_responses, support, witness):
    """Represent a mixed strategy by exact best-response linear constraints.

    payoffs is indexed by opponent actions, then our supported action indices;
    each coefficient row means base-minus-deviation expected payoff.
    """
    base = best_responses[0]
    supported = set(best_responses)
    equalities = []
    inequalities = []
    for action in range(len(payoffs)):
        if action == base:
            continue
        record = {
            'deviation_action': action,
            'coefficients_exact': [str(payoffs[base][j] - payoffs[action][j])
                                   for j in support],
        }
        (equalities if action in supported else inequalities).append(record)
    return {
        'variable_actions': list(support),
        'other_action_probabilities': '0',
        'sum_probabilities': '1',
        'probability_bounds': 'strictly_positive',
        'reference_best_response_action': base,
        'equalities_dot_coefficients_eq_zero': equalities,
        'inequalities_dot_coefficients_ge_zero': inequalities,
        'witness_probabilities_exact': list(witness),
    }


def solve_exact_nash_set(a, b, *, max_pairs=225, max_operations=100000,
                         max_actions=4, include_vertices=False,
                         max_vertex_candidates=50000):
    """Represent every Nash profile (up to 4x4, opt-in 5x5).

    Relatively open rational H-polytopes describe continuous equilibrium
    families; exact probability strings are authoritative over float previews.
    Budget exhaustion raises instead of asserting incomplete work is complete.
    """
    feasibility = solve_exact_support(a, b,
                                      max_pairs=max_pairs,
                                      max_operations=max_operations,
                                      max_actions=max_actions)
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    af = [[Fraction(str(v)) for v in row] for row in aa]
    bf = [[Fraction(str(v)) for v in row] for row in bb]
    bt = [list(row) for row in zip(*bf)]
    if type(include_vertices) is not bool or type(max_vertex_candidates) is not int or not 1 <= max_vertex_candidates <= 200000:
        raise ValueError("Invalid vertex options")
    families = []
    vertex_budget = {"limit": max_vertex_candidates, "tested": 0}
    for equilibrium in feasibility['equilibria']:
        I = equilibrium['support']['row']
        J = equilibrium['support']['col']
        row = _player_polytope(bt, J, I,
                               [equilibrium['row_probabilities_exact'][i] for i in I])
        col = _player_polytope(af, I, J,
                               [equilibrium['col_probabilities_exact'][j] for j in J])
        if include_vertices:
            row_geometry = enumerate_polytope_vertices(row, vertex_budget)
            col_geometry = enumerate_polytope_vertices(col, vertex_budget)
            row.update(row_geometry)
            col.update(col_geometry)
        family = {
            'support': {'row': list(I), 'col': list(J)},
            'row_strategy_polytope': row,
            'col_strategy_polytope': col,
        }
        if include_vertices:
            family['product_affine_dimension'] = (row['affine_dimension'] + col['affine_dimension'])
            family['product_closure_vertex_count'] = (len(row['closure_vertices_exact']) * len(col['closure_vertices_exact']))
        families.append(family)
    return {
        'equilibrium_set_complete': True,
        'exact_support_feasibility_complete': True,
        'all_equilibria_enumerated_as_points': False,
        'representation': 'disjoint union of strict-support Cartesian products of exact rational H-polytopes',
        'inequality_semantics': 'dot(coefficients_exact, supported probabilities) >= 0; equalities == 0',
        'input_semantics': feasibility['exact_input_model'],
        'support_pairs_checked': feasibility['support_pairs_checked'],
        'nonempty_support_cells': len(families),
        'exact_work_operations': feasibility['exact_work_operations'],
        'equilibrium_families': families,
        'vertices_computed': include_vertices,
        'vertex_candidates_tested': vertex_budget['tested'],
        'note': 'Every Nash profile belongs to exactly one strict-support cell. '
                'Cells may contain infinitely many strategies; each cell is specified '
                'by exact rational constraints, rather than sampled points.',
    }
