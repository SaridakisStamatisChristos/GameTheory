# Exact Nash-equilibrium families — mathematical contract

## Scope

For a finite two-player normal-form game with rational payoff matrices
\(A\in\mathbb{Q}^{m\times n}\), \(B\in\mathbb{Q}^{m\times n}\), the
`/api/equilibria/exact-nash-set` endpoint returns a **complete, exact
constraint representation** of the Nash set for \(1\le m,n\le4\), provided
the bounded computation finishes. It does **not** list infinitely many
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

- The default is at most **225 support pairs** and **100,000 counted
  elimination operations** (configurable within published bounds).
  Exact rational coefficient growth and memory use may still be expensive.
- The endpoint does **not** compute extreme points, dimensions, adjacency,
  equilibrium selection, or the total number of continuum points.
- The algorithm is exact under its stated input model; the mathematical
  proof establishes the description, while implementation correctness
  remains subject to testing, independent audit and possible defects.
- Tests check a direct rational Nash predicate against reported-cell
  membership for exhaustive small games, seeded 3×3 / 4×4 games,
  degenerate continua, strict-support boundaries, and HTTP failures.
  Existing LP and CFR solvers remain separate.
