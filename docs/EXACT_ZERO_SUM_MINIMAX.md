# Exact rational minimax solver (v51)

## Model and guarantee

Input is a finite, two-player **zero-sum** normal-form game. The only
payoff matrix is the row player's A; the column player's matrix is -A.
After validating finite float64 entries, each value is interpreted as
Fraction(str(float_value)), the same input contract as other exact
GameTheory algorithms. The output contains one exact rational saddle
point (x,y), value v, and the independently checked identity:

    min_j sum_i x_i A_ij = v = max_i sum_j A_ij y_j

It does not enumerate every Nash equilibrium. A complete-set claim would
be incorrect when optimal strategies form a continuum.

## Algorithm and correctness argument

Let t = 1 - min(A), M = A + t. Every element of M is at least 1.
Using rational variables u in R^m and z in R^n, solve these exact
standard-form dual linear programs:

    row LP: minimize   sum_i u_i
            subject to sum_i M_ij u_i >= 1  (all columns j)
                       u_i >= 0

    col LP: maximize   sum_j z_j
            subject to sum_j M_ij z_j <= 1  (all rows i)
                       z_j >= 0

Both are feasible and bounded because M is strictly positive. The
strong duality theorem establishes equality of their nonzero optimal
objective values S. Therefore x=u/S, y=z/S, and v=1/S-t. These are
maximin/minimax strategies for A. This code separately solves both LPs
via the shared bounded rational simplex and verifies, *from original
rational input payoffs*, that simplex normalization, nonnegativity,
LP objective equality, min/max security values, and bilinear expected
value agree exactly. A solver implementation error cannot be turned
into a silent success if these identities fail.

The certificate is for the mathematical equilibrium result on the
declared rational input model. There is no formally machine-verified
proof of code correctness and no independent audit of the simplex
algorithm. Worst-case simplex iterations and rational coefficient bit
growth can be exponential. The configured pivot-tableau-update counter
does not cap bit complexity or elapsed time.

## Service and reproduction

POST /api/zero-sum/exact accepts {"payoffs": [[...]], "max_operations": 200000}.
The existing 20-action-per-side validation remains, with an additional
100-action-profile exact LP ceiling. max_operations is 1..2,000,000.
Budget failures return HTTP 422, not an equilibrium response.

Run from backend:

    python -m pytest -q -W error test_exact_minimax_v51.py test_exact_minimax_http_v51.py
    python -m benchmark_exact_minimax_v51

The test suite exhausts all 81 entries in {-1,0,1}^4 for 2x2 payoffs,
tests seeded integer rectangular/square games up to 8x8, and compares
game value against an *independently formulated* SciPy/HiGHS maximization
LP directly on original payoffs. It also tests high dynamic ranges
and faults the LP outputs to confirm certificate rejection. The
benchmark reports medians, never a SOTA speedup claim.

## Prior art and next acceptance gates

This is a well-established LP reduction, **not a novel game-theory
algorithm**. Gambit's gambit-lp solves constant-sum games in rational
arithmetic by default:
https://gambitproject.readthedocs.io/en/latest/tools.lp.html
OpenSpiel provides a zero-sum matrix-game LP implementation:
https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/lp_solver.py

Before claiming competitive state-of-the-art, compare identical rational
inputs with gambit-lp, pinned versions and budgets, and include
representative ill-conditioned, degenerate, and larger games. Record
runtime distribution, bit growth, memory, and failure rates. Exact
zero-sum solving does **not** solve general-sum Nash equilibrium
computation or exponentially many support-family enumeration.
