"""Bounded exact rational phase-I/II simplex for Nash support feasibility.

Maximize the common lower bound on supported probabilities, with best-response
inequalities and equalities. No numerical tolerances are used. The operation
counter bounds tableau updates and exact coefficient inspections,
not the bit-complexity or wall-clock time. Optional tableau bit-length
tracking is opt-in and does not change the feasibility procedure.
"""
from fractions import Fraction as F
from exact_support import ExactWorkLimit


class _RationalSimplex:
    """Maximize c*x subject to A*x <= b and x >= 0."""

    def __init__(self, A, b, c, work):
        self.m, self.n = len(b), len(c)
        self.B = [self.n+i for i in range(self.m)]
        self.N = list(range(self.n)) + [-1]
        self.D = [[F(0) for _ in range(self.n+2)] for _ in range(self.m+2)]
        self.work = work
        for i in range(self.m):
            for j in range(self.n):
                self.D[i][j] = F(A[i][j])
            self.D[i][self.n] = F(-1)
            self.D[i][self.n+1] = F(b[i])
        for j in range(self.n):
            self.D[self.m][j] = -F(c[j])
        self.D[self.m+1][self.n] = F(1)
        if self.work.get('track_rational_bits', False):
            self._record_tableau_bits()

    def _record_tableau_bits(self):
        """Opt-in exact numerator/denominator growth diagnostics, not memory use."""
        numerators = (abs(value.numerator).bit_length()
                      for row in self.D for value in row)
        denominators = (value.denominator.bit_length()
                        for row in self.D for value in row)
        self.work['max_tableau_numerator_bits'] = max(
            self.work.get('max_tableau_numerator_bits', 0), max(numerators))
        self.work['max_tableau_denominator_bits'] = max(
            self.work.get('max_tableau_denominator_bits', 0), max(denominators))

    def _pivot(self, r, s):
        m, n, D = self.m, self.n, self.D
        self.work['steps'] += (m+2)*(n+2)
        if self.work['steps'] > self.work['limit']:
            raise ExactWorkLimit('Exact rational simplex operation budget exceeded')
        inv = F(1)/D[r][s]
        for i in range(m+2):
            if i == r:
                continue
            factor = D[i][s]*inv
            for j in range(n+2):
                if j != s:
                    D[i][j] -= D[r][j]*factor
        for j in range(n+2):
            if j != s:
                D[r][j] *= inv
        for i in range(m+2):
            if i != r:
                D[i][s] *= -inv
        D[r][s] = inv
        self.B[r], self.N[s] = self.N[s], self.B[r]
        if self.work.get('track_rational_bits', False):
            self._record_tableau_bits()

    def _phase(self, phase):
        m, n, D = self.m, self.n, self.D
        row = m+1 if phase == 1 else m
        while True:
            eligible = [j for j in range(n+1)
                        if D[row][j] < 0 and not (phase == 2 and self.N[j] == -1)]
            if not eligible:
                return True
            # Bland's entering-variable rule prevents degeneracy cycling.
            entering = min(eligible,key=lambda j:self.N[j])
            leaving = None
            for i in range(m):
                if D[i][entering] <= 0:
                    continue
                if (leaving is None or
                    (D[i][n+1]/D[i][entering],self.B[i]) <
                    (D[leaving][n+1]/D[leaving][entering],self.B[leaving])):
                    leaving = i
            if leaving is None:
                return False
            self._pivot(leaving, entering)

    def solve(self):
        m, n, D = self.m, self.n, self.D
        if m:
            minrhs = min(range(m),key=lambda i:(D[i][n+1],self.B[i]))
            if D[minrhs][n+1] < 0:
                self._pivot(minrhs,n)
                if not self._phase(1) or D[m+1][n+1] < 0:
                    return None
                if D[m+1][n+1] != 0:
                    return None
                if -1 in self.B:
                    i = self.B.index(-1)
                    j = min(range(n+1),key=lambda s:(D[i][s],self.N[s]))
                    self._pivot(i,j)
        if not self._phase(2):
            raise AssertionError('Strict-support LP unexpectedly unbounded')
        x = [F(0)]*n
        for i,v in enumerate(self.B):
            if v < n:
                x[v] = D[i][n+1]
        return x


def feasible_strict_strategy(payoff, best_responses, support, work, *, fast_path=True, sign_screen=True):
    """Find an exact positive support mixture or report infeasibility.

    A uniform exact rational witness is checked *before* the simplex.
    For singleton supports it is also a complete infeasibility test:
    the only legal supported strategy is (1). For larger supports, an
    optional sign screen proves infeasibility when a required payoff tie
    or best-response inequality is impossible for *strictly positive*
    probabilities. Inconclusive supports fall back to the exact LP.
    fast_path=False preserves the unchanged simplex-only oracle.

    Payoff rows are pure actions, columns are opponent actions. Strategies
    are positive only on support; all best_responses must be tied at the
    highest payoff, including comparisons to unsupported pure actions.
    """
    k = len(support)
    if type(fast_path) is not bool:
        raise ValueError('fast_path must be boolean')
    if type(sign_screen) is not bool:
        raise ValueError('sign_screen must be boolean')
    if fast_path:
        # This exact row-sum comparison is equivalent to checking an
        # expected payoff at the uniform mixture: the common 1/k factor
        # cancels. Work is charged for every inspected matrix coefficient.
        # Crucially, it is charged even on unsuccessful probes, preserving
        # fail-closed operation budgets and avoiding free computation.
        probe_cost = len(payoff)*k
        work['steps'] += probe_cost
        work['probe_operations'] = work.get('probe_operations', 0) + probe_cost
        if work['steps'] > work['limit']:
            raise ExactWorkLimit('Exact feasibility probe operation budget exceeded')
        base = best_responses[0]
        expected_total = sum(payoff[base][j] for j in support)
        prescribed = set(best_responses)
        uniform_valid = True
        for i, row in enumerate(payoff):
            total = sum(row[j] for j in support)
            if total > expected_total or (i in prescribed and total != expected_total):
                uniform_valid = False
                break
        if uniform_valid:
            work['uniform_witness_hits'] = work.get('uniform_witness_hits', 0) + 1
            return [F(1, k)]*k
        if k == 1:
            # Proof: no other strictly positive probability vector exists.
            work['singleton_infeasible_hits'] = work.get('singleton_infeasible_hits', 0) + 1
            return None
        if sign_screen:
            # For any strictly positive support mixture p, a nonzero
            # coefficient vector with one weak sign has a strict signed
            # dot product. Such a vector cannot satisfy a required tie;
            # a weakly negative, nonzero base-minus-deviation vector
            # cannot satisfy its best-response inequality either.
            # Charge the *entire* exact sign pass even on early rejection.
            screen_cost = (len(payoff)-1)*k
            work['steps'] += screen_cost
            work['sign_screen_operations'] = work.get('sign_screen_operations', 0) + screen_cost
            if work['steps'] > work['limit']:
                raise ExactWorkLimit('Exact sign-screen operation budget exceeded')
            base_payoff = payoff[base]
            for i, row in enumerate(payoff):
                if i == base:
                    continue
                diffs = [base_payoff[j]-row[j] for j in support]
                any_positive = any(value > 0 for value in diffs)
                any_negative = any(value < 0 for value in diffs)
                if ((i in prescribed and any_positive != any_negative) or
                        (i not in prescribed and any_negative and not any_positive)):
                    work['sign_screen_rejections'] = work.get('sign_screen_rejections', 0) + 1
                    return None
    work['simplex_lp_calls'] = work.get('simplex_lp_calls', 0) + 1
    constraints, rhs = [], []

    def add(row,limit):
        constraints.append([F(z) for z in row])
        rhs.append(F(limit))

    add([1]*k+[0],1)
    add([-1]*k+[0],-1)
    for j in range(k):
        add([-int(j==i) for i in range(k)]+[1],0)
    add([0]*k+[1],1)
    base = best_responses[0]
    br = set(best_responses)
    for i in range(len(payoff)):
        if i == base:
            continue
        diff = [payoff[base][j]-payoff[i][j] for j in support]
        add([-z for z in diff]+[0],0)
        if i in br:
            add(diff+[0],0)
    solution = _RationalSimplex(constraints,rhs,[F(0)]*k+[F(1)],work).solve()
    if solution is None or solution[-1] <= 0:
        return None
    p = solution[:k]
    if sum(p) != 1 or any(z <= 0 for z in p):
        raise AssertionError('Invalid rational strict-support solution')
    base_value = sum(payoff[base][j]*z for j,z in zip(support,p))
    for i in range(len(payoff)):
        value = sum(payoff[i][j]*z for j,z in zip(support,p))
        if value > base_value or (i in br and value != base_value):
            raise AssertionError('Strict-support LP failed best-response verification')
    return p
