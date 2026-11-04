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
2. **Independent implementation reference.** Compare representative
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
4. **Game-family coverage.** Report success and regret distributions on
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
