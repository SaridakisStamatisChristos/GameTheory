"""Support-pair LP feasibility Nash solver for two-player finite games.

Enumerates unequal support sizes and degenerate games. A single representative
per support pair is returned; continuous equilibrium sets are not enumerated.
"""
import itertools
from fractions import Fraction
from safe_numeric import expected_payoffs, json_number
import numpy as np
from scipy.optimize import linprog


def solve_support_lp(a, b, max_pairs=100000, tol=1e-8):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or b.shape != a.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected equally shaped, nonempty finite payoff matrices')
    m, n = a.shape
    pairs = (2**m-1)*(2**n-1)
    if pairs > max_pairs:
        raise ValueError(f'{pairs} support pairs exceed max_pairs={max_pairs}')
    found = []
    # Numerical LP support enumeration does not prove support feasibility, even
    # when every combinatorial support pair was visited.
    normalized_a = np.asarray((np.asarray(a,dtype=np.longdouble)-np.min(a)) / (np.longdouble(np.max(a))-np.longdouble(np.min(a)) or np.longdouble(1)),dtype=float)
    normalized_b = np.asarray((np.asarray(b,dtype=np.longdouble)-np.min(b)) / (np.longdouble(np.max(b))-np.longdouble(np.min(b)) or np.longdouble(1)),dtype=float)
    for ki in range(1,m+1):
        for I in itertools.combinations(range(m),ki):
            for kj in range(1,n+1):
                for J in itertools.combinations(range(n),kj):
                    # Solve for each player's strategy independently; constraints
                    # require opponent-supported actions to be best responses.
                    # Positive support enforced by maximizing minimum probability t.
                    def feasible(matrix, supported, other):
                        k=len(other)
                        # matrix rows are actions, columns are opponent support.
                        # Variables: opponent support probabilities and t.
                        eq=[np.r_[np.ones(k),0.]]; rhs=[1.]
                        base=supported[0]
                        for i in supported[1:]:
                            eq.append(np.r_[matrix[i,other]-matrix[base,other],0.]);rhs.append(0.)
                        ub=[]
                        for i in range(matrix.shape[0]):
                            if i not in supported:
                                ub.append(np.r_[matrix[i,other]-matrix[base,other],0.])
                        for j in range(k):
                            v=np.zeros(k+1);v[j]=-1;v[-1]=1;ub.append(v)
                        res=linprog(np.r_[np.zeros(k),-1.], A_ub=np.asarray(ub) if ub else None,
                            b_ub=np.zeros(len(ub)) if ub else None,
                            A_eq=np.asarray(eq),b_eq=np.asarray(rhs),
                            bounds=[(0,None)]*k+[(0,None)],method='highs')
                        return res.x[:k] if res.success and res.x[-1]>0 else None
                    # row best-response constraints determine y over J.
                    y0=feasible(normalized_a,I,J)
                    if y0 is None:continue
                    # column best-response constraints determine x over I.
                    x0=feasible(normalized_b.T,J,I)
                    if x0 is None:continue
                    x=np.zeros(m);y=np.zeros(n);x[list(I)]=x0;y[list(J)]=y0
                    # Certified in each player's own normalized payoff units.
                    gap=max(float(np.max(normalized_a@y)-x@normalized_a@y),
                            float(np.max(x@normalized_b)-x@normalized_b@y),0.)
                    if gap>1e-8:continue
                    if any(np.max(np.abs(x-np.array(z['row_probabilities'])))<tol and
                           np.max(np.abs(y-np.array(z['col_probabilities'])))<tol for z in found):continue
                    found.append({'row_probabilities':x.tolist(),'col_probabilities':y.tolist(),
                                  'expected_payoffs':expected_payoffs(a,b,x,y),
                                  'nash_gap':gap,'nash_gap_units':'per-player normalized',
                                  'support':{'row':list(I),'col':list(J)}})
    if (m,n)==(2,2):
        # Exact rational indifference oracle for the unique interior 2x2 profile.
        # It catches strictly positive probabilities smaller than an LP solver's
        # absolute feasibility threshold (e.g., probability 1e-10).
        af=[[Fraction(str(z)) for z in row] for row in a]
        bf=[[Fraction(str(z)) for z in row] for row in b]
        denominator_q=af[0][0]-af[0][1]-af[1][0]+af[1][1]
        denominator_p=bf[0][0]-bf[1][0]-bf[0][1]+bf[1][1]
        if denominator_q and denominator_p:
            q=(af[1][1]-af[0][1])/denominator_q
            pp=(bf[1][1]-bf[1][0])/denominator_p
            if 0<pp<1 and 0<q<1:
                x=np.array([float(pp),float(1-pp)]);y=np.array([float(q),float(1-q)])
                if not any(np.array_equal(x,np.asarray(z['row_probabilities'])) and
                           np.array_equal(y,np.asarray(z['col_probabilities'])) for z in found):
                    found.append({'row_probabilities':x.tolist(),
                                  'col_probabilities':y.tolist(),
                                  'expected_payoffs':expected_payoffs(a,b,x,y),
                                  'nash_gap':0.,'nash_gap_units':'exact rational input',
                                  'support':{'row':[0,1],'col':[0,1]},
                                  'certificate':'exact rational indifference'})
    return {'equilibria':found,'support_pairs_checked':pairs,
            'complete_support_coverage':True,
            'all_feasible_supports_resolved':False,
            'note':'All support pairs enumerated, but numeric LP feasibility is not a completeness proof. Exact rational interior 2x2 witness included when present; continuum equilibria not fully characterized.'}
