# v55 — Certified rational feasibility fast paths

## Exact mathematical contract

Let `J` be a nonempty opponent-action support of size k and `I` the
nonempty set of prescribed best responses. The strict-support Nash
feasibility query asks for a positive probability vector over J, summing
to 1, such that every action in I is a best response and no action
outside I strictly exceeds them.

**Uniform witness fast path.** Set p_j=1/k for all j in J. For every
action i, compute the *exact rational row sum* s_i = sum_{j in J} A_ij.
If s_i <= s_base for every i and s_i=s_base for all i in I, then p is
a valid *strictly positive* feasible witness. Dividing the sums by k
does not change comparisons. This is a sufficient condition, never
a necessary one when k>1.

**Singleton complete path.** If k=1, the uniform vector (1) is the
*only* admissible strict-support strategy, so failure of the row-sum
test is an exact proof of infeasibility. Thus no simplex call is needed
for any singleton opponent support.

**Fallback.** For k>1, an unsuccessful uniform probe invokes the
unchanged two-phase, Bland-rule rational simplex and independently
rechecks its returned best-response properties. The uniform probe
never claims an unsupported Nash equilibrium. All returned full
profiles still pass the existing exact rational independent verifier.
The previous algorithm remains callable with
`feasibility_fast_path=False`, enabling exact matched baselines.

## Budget and measurement semantics

The existing rational simplex work counter charges every pivot's
tableau cell updates. v55 **also charges `len(payoff)*k`** units for
each exact row-sum coefficient probe, including failed probes,
before performing any coefficient arithmetic. Exceeding the work
budget raises `ExactWorkLimit` and cannot create a false complete
search. The returned metadata distinguish probe operations, uniform
hits, singleton-infeasible eliminations, and simplex LP calls.
The counter is a deterministic *work proxy*, **not** a bit-complexity,
memory, or wallclock guarantee. Changing this accounting alters
the number of support pairs reachable under the same operation
budget; completion flags remain authoritative.

## Evaluation

Run from backend:

    python -m pytest -q -W error test_exact_probe_v55.py
    python benchmark_exact_probe_v55.py --repetitions 3

The benchmark compares identical ordered support pairs, matrices,
maximum exact work, maximum support cardinality, and pruning mode
(pruning disabled in both arms to isolate feasibility). It alternates
baseline/optimized execution order, reports *all* wallclock samples,
medians, per-arm LP calls and exact operations, and checks every
returned equilibrium against independent exact Nash conditions.
It includes zero-tie/degenerate, dominance, coordination, cyclic,
random, and larger rectangular support-prefix workloads.

Some distributions may regress due to unsuccessful probe overhead.
A few CI repetitions cannot justify a global speedup or SOTA claim.
The existing external Gambit/OpenSpiel comparison gates continue
unchanged.

## Open limitations

- No fractional-integer/GCD representation optimization yet.
- Fraction numerator/denominator bit growth and peak memory remain
  unbounded by the work counter.
- Simultaneous-game general-sum exact Nash search remains exponential
  in the worst case; this does not change its completeness theory.
- The probe computes row sums in rational arithmetic; future adaptive
  selection should be evidence-validated rather than tuned to CI alone.
