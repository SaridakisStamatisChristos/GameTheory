# v60 — Certified polynomial Nash decomposition for 3-player binary games

The GameTheory research engine gains a **deterministic, exact algebraic
mixed Nash equilibrium solver** in addition to v59's numerical,
incomplete multistart support search.

Endpoint: POST /api/multiplayer/mixed/polynomial-exact

Request:

~~~json
{
  "payoffs": [
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
  ],
  "max_cells": 27
}
~~~

Each tensor is [action of player 0][action of player 1]
[action of player 2], with the outer list ordered by player.
Payoffs must be finite validated float64, interpreted thereafter
as exact decimal-string rational values. Exactly three players
and exactly two actions each are required. Other v59 multiplayer
shapes remain available through the older numerical endpoint.

## Exact support decomposition

Each player either plays pure action 0, pure action 1, or
mixes strictly positively over both. These three possibilities
give exactly 3^3 = **27 disjoint support cells**, covering
*every* strategy profile. A full equilibrium set is certified
only when each support cell has been classified:

- **Zero mixed players:** check all eight pure profiles by
  exact rational unilateral-deviation inequalities.
- **One mixed player:** its required indifference is constant.
  The other two players impose affine inequalities on its mixing
  probability. The exact Nash cell is either empty, a rational
  isolated point, or a rational interval with correctly marked
  open support boundaries and closed internal incentive boundaries.
- **Two mixed players:** each supported indifference is affine
  in the other mixed player's probability. A nondegenerate
  equation fixes that probability rationally; a degenerate zero
  equation leaves it free. The pure player's best-response
  inequality is bilinear in the remaining free probabilities.
  Its maximum over the closed probability cube is attained at
  a corner. If its maximal corner value is strictly positive,
  an interior point satisfies it by continuity. If the maximum
  is zero, an interior point is feasible only when *every* corner
  is zero, since a bilinear polynomial is a strictly positive
  convex combination of all corner values at interior points.
  Thus each nonempty continuous cell receives an **exact rational
  semialgebraic inequality representation**, not guessed samples.
- **Three mixed players:** for x=P1(action1), y=P2(action1),
  z=P3(action1), the indifference equations are
  F0(y,z)=F1(x,z)=F2(x,y)=0 with rational bilinear coefficients.

Write

    F0(y,z) = (a0+a2*z) + (a1+a3*z)*y
    F1(x,z) = (b0+b2*z) + (b1+b3*z)*x
    F2(x,y) = c0+c1*x+c2*y+c3*x*y.

When the relevant denominators are nonzero, eliminate
x=Nx(z)/Dx(z), y=Ny(z)/Dy(z) and substitute into F2:

    P(z) = c0*Dx*Dy + c1*Nx*Dy + c2*Ny*Dx + c3*Nx*Ny.

**P has degree at most 2.** The engine enumerates all real
roots of P using exact rational arithmetic and a quadratic
field Q(sqrt(discriminant)); perfect-square discriminants
reduce to rational roots. Every candidate requires 0<x,y,z<1
and is independently audited against *original payoff tensors*
using exact algebraic pure-action expectations. Results include
a machine-readable exact rational or algebraic representation,
not just floating point approximations.

If P is identically zero, a denominator vanishes identically,
or P admits an interior singular root where a cancelled
indifference numerator also vanishes, the exceptional
fully-mixed support is **unresolved**. The solver returns a
partial decomposition and the exact indifference coefficient
system for further work. It does not claim that all
equilibria were enumerated, even if other support cells
were solved.

## Semantics and certificates

The response distinguishes:

- isolated_equilibria: **verified exact algebraic points**;
- continuous_equilibrium_cells: complete rational
  interval or bilinear semialgebraic descriptions of those
  nonempty lower-dimensional support cells;
- unresolved_support_cells: exceptional fully-mixed systems
  not yet classified, with explicit reasons;
- complete_nash_set=true only when all 27 positive-support
  cells were visited and there are no unresolved systems.

An algebraic number is serialized as
type="quadratic", with exact rational strings a, b,
radicand denoting a+b*sqrt(radicand).
A rational probability is serialized with
type="rational", value="numerator/denominator".
These are mathematical exactness claims **within the declared
float64-decimal input model**, not on arbitrary-precision
unvalidated JSON tokens.

max_cells (1..27) is a separate strict support-classification
budget. If it is exhausted, unvisited cells remain unknown
and full completeness is forbidden. This bound is not a
bit-complexity guarantee; rational numerator growth and
wall-clock/memory costs are separate.

## Verification

- Known quadratic systems with two interior **irrational**
  fully-mixed roots, one rational root, a double root,
  negative discriminant and linear eliminant.
- Full exact support classification including pure, one-mixed
  interval, two-mixed isolated and bilinear regions;
  strict positivity and open/closed interval boundaries.
- Genuine three-player parity degeneracy and singular
  denominator counterexample return explicitly partial.
- An independent field-pair payoff audit recomputes all
  three players' original-payoff supported indifference
  identities exactly for the irrational example.
- Fault, invalid dimension, missing-budget and API isolation
  tests. A reproducible 18-game seeded evidence campaign
  checks completeness semantics, known root counts, direct
  numerical original-payoff regrets, and timing distributions.

The v60 GitHub Actions benchmark is archived as
multiplayer-polynomial-v60-evidence. The previous
v49–v59 correctness and CI gates remain active.

## Non-claims and future work

The algorithm is **not** a general N-player polynomial
system solver and it is not a complete isolated-equilibrium
enumerator for larger games. It is complete on the
3-player/2-action subclass **only when the returned
complete_nash_set flag is true**. Singular fully-mixed systems
may contain algebraic points or continuous varieties that
this release does not resolve. The continuous-cell output
is an exact constraint representation, not a list of its
infinitely many points.

No polynomial-time general Nash algorithm, formal proof
assistant certification, external symbolic engine equivalence
or global SOTA speed ranking is claimed.
