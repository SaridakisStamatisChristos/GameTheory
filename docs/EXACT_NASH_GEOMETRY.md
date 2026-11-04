# Exact Nash-equilibrium families — mathematical contract

## Scope

For a finite two-player normal-form game with rational payoff matrices
\(A\in\mathbb{Q}^{m\times n}\), \(B\in\mathbb{Q}^{m\times n}\), the
`/api/equilibria/exact-nash-set` endpoint returns a **complete, exact
constraint representation** of the Nash set for \(1\le m,n\le4\), provided
the bounded computation finishes. The default limit is 4×4 and an explicit
`max_actions=5`, `max_pairs=961` budget enables 5×5 games. It does **not** list infinitely many
equilibrium points.

The payoff rationals are \(`Fraction(str(float(value)))`\) of the validated
Python float64 values. Consequently this is exact for the validated-decimal
game, not arbitrary-precision original JSON tokens.

## A constructive finite decomposition

Let \(I\subseteq\{0,\ldots,m-1\}\) and
\(J\subseteq\{0,\ldots,n-1\}\) be **nonempty** supports.
Fix \(i_0\in I\) and \(j_0\in J\). Define two sets:

```text
Q(I,J) = { y : y[j] > 0 for j in J, y[j] = 0 outside J,
            sum(y) = 1,
            (A[i0,:] - A[i,:]) dot y = 0 for every i in I,
            (A[i0,:] - A[k,:]) dot y >= 0 for every k outside I }

P(I,J) = { x : x[i] > 0 for i in I, x[i] = 0 outside I,
            sum(x) = 1,
            x dot (B[:,j0] - B[:,j]) = 0 for every j in J,
            x dot (B[:,j0] - B[:,l]) >= 0 for every l outside J }
```

All coefficient vectors are exact rational strings. A cell is the
Cartesian product \(P(I,J)\times Q(I,J)\). Each factor is the
strict-support portion of a rational polytope.

**Theorem (exact Nash-set decomposition).** The Nash set equals the disjoint
union of the nonempty cells \(P(I,J)\times Q(I,J)\) over all nonempty
support pairs \((I,J)\).

**Proof.** If \((x,y)\) is a Nash equilibrium, choose its strictly positive
supports \(I\) and \(J\). Every supported pure action of a player must
attain that player's maximal expected payoff, and every unsupported action
must yield no more. Thus the equalities, weak inequalities, strict
positivity, and simplex equations in the corresponding cell hold.
Conversely, if a strategy pair belongs to a cell, all actions assigned
positive probability are best responses to the opponent, so neither
player can profit by deviating. Hence it is a Nash equilibrium. Strictly
positive supports are unique, so distinct cells do not overlap. QED.

The existing `exact_support.py` solver constructs one strict rational
witness per feasible support pair using bounded Fourier–Motzkin elimination.
The family endpoint reuses that enumeration, then supplies all best-response
constraints rather than equating the single witness with the entire set.
If exact feasibility exhausts a declared work budget, it returns HTTP 422,
never a partial response marked complete.

## Reading the endpoint's output

For each `equilibrium_families` entry:

- `support.row` and `support.col` are the exact positive-support indices;
- `row_strategy_polytope` describes the row player's \(x\) using
  column-payoff matrix \(B\); `col_strategy_polytope` describes \(y\)
  using row-payoff matrix \(A\);
- `variable_actions` identifies the positive variables, all other
  probabilities being zero; their sum is one;
- each `equalities_dot_coefficients_eq_zero` vector has dot product
  exactly zero with those variables;
- each `inequalities_dot_coefficients_ge_zero` vector has dot product
  greater than or equal to zero;
- `witness_probabilities_exact` supplies one **exact** positive witness,
  not a complete enumeration.

Example: for a 3×3 zero-payoff game there are
\((2^3-1)^2=49\) nonempty strict-support cells, whose union covers the
whole product of two probability simplices. Many cells contain infinitely
many equilibria.

## Limitations and verification

- By default, at most **225 support pairs** and **100,000 counted
  elimination operations** are processed (maximum 500,000 in default mode).
  Opt-in 5×5 permits up to **961 support pairs** and **2,000,000** counted
  elimination operations. Neither is a polynomial-time guarantee.
  Exact rational coefficient growth and memory use may still be expensive.
- With `include_vertices=true`, the endpoint computes **all exact vertices**
  and **affine dimensions** of each factor-polytope *closure*. A vertex can lie
  on an excluded strict-support boundary, so it need not belong to that cell.
  The `product_affine_dimension` is the sum of row/column factor dimensions;
  `product_closure_vertex_count` is their vertex-count product.
  This is not a global adjacency or connected-component decomposition.
  `max_vertex_candidates` defaults to 50,000 and is capped at 200,000.
- The algorithm is exact under its stated input model; the mathematical
  proof establishes the description, while implementation correctness
  remains subject to testing, independent audit and possible defects.
- Tests check a direct rational Nash predicate against reported-cell
  membership for exhaustive small games, seeded 3×3 / 4×4 games,
  degenerate continua, strict-support boundaries, and HTTP failures.
  Existing LP and CFR solvers remain separate.

## v44 exact closure geometry — derivation

For each nonempty support cell `P(I,J) × Q(I,J)`, the exact-support
solver supplies a witness whose active probabilities are **all strictly
positive**. Let `C` be the weak-constraint polytope obtained from one
factor by replacing all `p_i > 0` with `p_i >= 0` while preserving the
best-response equalities, inequalities, and simplex equation.

Then the closure of that factor is **exactly** `C`: every strict-support
strategy lies in `C`, and for any `z in C`, interpolating with the strict
witness `w` gives `(1-t)z + t w` with every supported coordinate
positive for `t>0`, converging to `z` as `t -> 0`.

Because `C` is bounded in the simplex and rational, it is the convex
hull of finitely many rational vertices. The implementation computes a
basis of independent equality constraints, considers subsets of
inequalities active at an extreme point, solves each full-rank square
system with `Fraction` Gauss–Jordan elimination, validates *all* exact
equalities/inequalities, and deduplicates the resulting vertices.

The rank of the difference vectors relative to one vertex equals the
affine dimension of `C`; the dimension of the Cartesian-product cell is
the sum of the two factor dimensions. Vertices of the **closure** need
not satisfy strict positivity for that cell (but still lie in some Nash
support cell of the same game).

The search is combinatorial, not scalable in general. The cumulative
candidate-basis limit prevents incomplete vertex lists from being
presented as complete. An exceeded budget returns an error, not a
partial claim of mathematical completeness. The v44 tests independently
compare random linear-objective maxima against SciPy/HiGHS optimizers,
exercise degenerate 5×5 simplex families, and validate strict-boundary
semantics.

## Faster exact feasibility for opt-in 5×5 games

General-support Fourier–Motzkin projection can generate a combinatorial
number of intermediate inequalities and exhaust its work allowance on
ordinary 5×5 matrices. The opt-in 5×5 mode instead uses **exact bounded
vertex-basis feasibility** on each player's weak best-response polytope.

For a nonempty bounded simplex polytope, every point is a convex
combination of vertices. A strictly positive strategy on a prescribed
support exists iff, for **every** support coordinate, at least one feasible
vertex assigns that coordinate positive mass. The arithmetic average
of all enumerated vertices then has strictly positive mass on every
coordinate if and only if the support is feasible. All coefficients,
basis solutions and resulting witnesses are rational. No numerical
tolerance is used for this classification.

If a weak polytope is infeasible or lacks positive mass for a required
support action, that support is rejected. Basis enumeration consumes the
same global `max_operations` budget and fails closed if exceeded.
This improves the opt-in 5×5 path without replacing the well-tested
default 4×4 Fourier–Motzkin implementation.

The independent v44 suite includes seeded nondegenerate 5×5 games,
a direct rational Nash-regret oracle for every returned witness,
and exact vertex inequality checks. The algorithms remain exponential
in general and no 6×6 completeness claim is made.
