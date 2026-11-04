# GameTheory — comparative SOTA evidence gates (v48 research campaign)

## Scope and non-claims

GameTheory targets a mathematically transparent **finite, two-player
normal-form** equilibrium workbench, including bounded exact rational
support cells and certified numerical candidates. The core is
dimension-independent, but exact exhaustive support enumeration is
exponential and the API enforces a local-service matrix limit.

**SOTA is not yet demonstrated.** Distinguish code correctness,
mathematical completeness under budgets, comparative runtime,
broader algorithmic coverage, and operational maturity.

## Measurable acceptance gates

1. **Correctness and contracts.** All returned exact support cells have
   independent rational Nash certificates. Complete-set claims require
   classifying every nonempty support (including soundly pruned pairs).
   A truncated rational budget must produce partial/inconclusive states.
   Exhaustive 2×2 and seeded degenerate 3×3/4×4 oracles must agree
   with unpruned search, and independent SciPy/HiGHS oracle tests must
   match conditional dominance classifications on bounded integer games.
2. **Independent implementation reference.** v49 adds a reproducible
   **Nashpy 0.0.43** support-enumeration comparison, with direct numerical
   unilateral-deviation certification on each reference candidate.
   Its `benchmark_independent_nashpy_v49.py` script checks 14 named
   and seeded 2×2/3×3 cases and reports point-vs-family output
   differences; exact enumeration is exhausted for these cases. This is
   a tested comparator, *not* evidence of superior runtime or broad SOTA.
   Next, compare representative
   nondegenerate equilibria with Gambit (`pygambit` or
   `gambit-enummixed`) using the **same rational payoffs**. Compare
   zero-sum and extensive-form algorithms with OpenSpiel only where
   models and metrics match. Preserve each tool's exact versions,
   environment, input fixtures, time budgets, and solver options.
   A tool's absence is *not* a PASS.
3. **Fair performance baseline.** Run a paired pure-pruned, unpruned,
   and mixed-pruned exact search over identical ordered support prefixes
   and equal total rational-work allowances. Measure preprocessing,
   LP attempts, exact work, elapsed time, and failures; report medians,
   dispersion, and sample size. Include tie-heavy and adversarial
   instances on which optional mixed dominance may regress.
   The script `backend/benchmark_mixed_dominance_v48.py` checks these
   equivalences before reporting times. One CI sample is a **smoke
   benchmark**, not a statistical ranking.
4. **Game-family coverage.** The v49 comparator includes matching
   pennies, battle of the sexes, prisoner's dilemma, rock-paper-scissors,
   3×3 coordination, all-zero degeneracy, and seeded random games.
   It does not replace a broad distributional benchmark with
   statistically meaningful uncertainty.
    Report success and regret distributions on
   matching pennies, coordination, rock-paper-scissors, battle-of-the-
   sexes, degenerate games, random square/rectangular games, and
   strategically dominated games. Record sizes, seeds, tolerance,
   timeout policy and number of missing/undecided results.
5. **General algorithmic capability.** Once correctness and baselines
   are established, test topology-aware Nash decomposition,
   stronger dominance/symmetry reductions, smarter support ordering,
   and exact rational LP improvements. Do not add hardcoded 6×6,
   7×7, etc. special cases or claim polynomial-time general Nash.
6. **Release evidence.** Exact branch/commit SHA, strict-warning
   Python suite, 99% statement/branch coverage gate, frontend build,
   reproducible benchmark artifact, and a rollback path. Browser-level
   integration tests remain a separately tracked capability gap.

## Evidence states

- **Implemented:** source and API exposed.
- **Locally or CI verified:** execution records on the exact commit.
- **Mathematically justified:** an explicit proof of a stated invariant
  (distinct from formal machine verification of its implementation).
- **Comparatively demonstrated:** fair, repeatable results versus
  independently maintained libraries on matching problem models.
- **Broader SOTA:** requires multiple current comparator families,
  credible runtime and numerical results, no unresolved critical
  correctness failures, and public reproducible evidence.

Never promote a lower state as if it proved the next one.

## External comparator documentation

- Gambit Nash equilibrium methods:
  https://github.com/gambitproject/gambit/blob/master/doc/gui.nash.rst
- Gambit exact two-player extreme equilibrium computation:
  https://gambitproject.readthedocs.io/en/v16.6.0-zobrist-ipl.2/tools.enummixed.html
- OpenSpiel algorithm inventory:
  https://openspiel.readthedocs.io/en/latest/algorithms.html

These are **references**, not performance results. No Gambit/OpenSpiel
comparative speedup, SOTA status, production deployment, or formal
verification claim is made on the strength of this document.

## v49 comparable evidence: Nashpy only

Nashpy's normal-form `support_enumeration` returns **numerical point
profiles**. GameTheory's exact search returns **strict-support H-cells**;
in degenerate games a cell may contain infinitely many profiles.
Therefore raw result counts are **not** directly comparable.

For each Nashpy point we independently check simplex normalization,
nonnegative probabilities, finite values, and original-payoff
unilateral regret. Only validated Nashpy points are considered when
checking that the exact solver found a containing support family.
The benchmark reports any invalid numerical profiles and warnings,
rather than treating them as exact witnesses.

The fixed-size comparison uses a **full exact support search** to
make support-coverage comparisons meaningful. Timing output is
descriptive only, includes different methods and output types, and
is not used to assert a cross-tool speedup.

External Gambit and OpenSpiel runs remain **not executed** in this
campaign. Their APIs and environments require separate pinned
cross-engine benchmarking; no results are invented.

## v50: exact correlated equilibrium (bounded rational LP)

The new `POST /api/correlated-equilibrium/exact` endpoint complements the existing
numerical correlated-equilibrium endpoint. It solves the finite two-player
recommendation-obedience LP with the **existing rational simplex**, using the
decimal-string interpretation of validated float64 payoffs. It returns one
**exact rational** joint distribution with direct probability and all
unconditional recommendation-obedience inequalities rechecked in rational
arithmetic (including zero-probability recommendations).

Options: `objective=welfare|feasible` and `max_operations=1..2000000`.
The `welfare` objective maximizes the **raw sum of the two players' expected
payoffs**, unlike the existing *numerical* endpoint's per-player-normalized
objective. The `feasible` mode selects an arbitrary feasible CE. The
exact service caps inputs at **36 joint action profiles**; the general API
retains its separate 20×20 input-validation ceiling. Exceeding the bounded
pivot-update-work budget returns HTTP 422, not a falsely successful certificate.
The budget does **not** bound coefficient bit growth or wall time.

`joint_distribution_exact`, `expected_payoffs_exact`, and
`objective_value_exact` are rational strings. Output correctness is checked
independently of the LP's constraints with Fraction arithmetic; a separately solved
exact rational dual LP supplies nonnegative multipliers; direct independent
rational checking of dual feasibility and matching primal/dual objectives
certifies the reported objective optimum. The returned distribution
is **one** correlated equilibrium, not a complete description of the CE
polytope or Nash equilibrium set.

`test_exact_correlated_v50.py` compares seeded 2×2, rectangular and 4×4
integer games with independently formulated SciPy/HiGHS welfare LPs, checks
all obedience inequalities directly, covers pathological inputs and budget
failures, and faults the LP outputs to verify that invalid certificates are
rejected. This is a meaningful coverage extension, **not** a claim of
cross-engine SOTA. Follow-up evidence gates: independently check LP dual
optimality, reproduce Gambit/OpenSpiel comparisons on matching models,
and examine bit-complexity/wall-clock behavior under adversarial rational data.

## v51: bounded exact zero-sum minimax with primal-dual certificate

The new `POST /api/zero-sum/exact` solves the classic strictly-positive
payoff translation dual LPs in rational arithmetic and directly verifies
original-unit maximin/minimax security values and dual objective equality.
It uses 100 total action profiles maximum and a bounded rational tableau
update counter; no bound on coefficient bit growth or elapsed time is implied.

The numeric `POST /api/zero-sum` endpoint remains untouched. The new
module supports exact one-point equilibrium certification for zero-sum
games without exponential *support-family enumeration*. It is not a
replacement for general-sum Nash enumeration, and one certified saddle
point does not imply a complete set of all equilibria.

`test_exact_minimax_v51.py` provides exhaustive small 2×2 3-valued
fixtures, seeded rectangular/square fixtures and independent direct
SciPy/HiGHS value comparisons, plus adversarial fault injection.
`benchmark_exact_minimax_v51.py` is a descriptive smoke comparator.
Gambit `gambit-lp` exact-rational is documented prior art and an
outstanding fair runtime comparator; OpenSpiel comparisons also remain
not executed. **No global SOTA claim follows.**
