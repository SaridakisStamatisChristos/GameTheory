"""General-size game-theory solver router with explicit evidence states.

Pure equilibria and strictly dominated strategy eliminations use exact
comparisons of validated binary64 inputs. General-sum mixed candidates are
numerical; exact equilibrium-set completeness requires full bounded support
search and exact rational verification, never an approximate heuristic.
"""
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
from approximate import approximate_nash
from independent_lh import solve_independent_lh
from numerical_certificate import certify_bimatrix
from general_exact_search import search_exact_nash_cells


def _validate(a,b):
    try:
        a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('Nonrectangular or nonnumeric payoffs') from exc
    if (a.ndim!=2 or a.size==0 or b.shape!=a.shape or
            not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Expected finite nonempty matching payoff matrices')
    return a,b


def eliminate_strict_pure_dominance(a,b):
    """Iterated pure strict dominance; never discards a Nash best response."""
    rows=list(range(a.shape[0]))
    cols=list(range(a.shape[1]))
    removed=[]
    while True:
        change=False
        for i in rows:
            winner=next((k for k in rows if k!=i and np.all(a[k,cols]>a[i,cols])),None)
            if winner is not None:
                rows.remove(i)
                removed.append({'player':'row','action':i,'dominated_by':winner,
                                'opponent_actions':cols.copy()})
                change=True
                break
        if change:
            continue
        for j in cols:
            winner=next((k for k in cols if k!=j and np.all(b[rows,k]>b[rows,j])),None)
            if winner is not None:
                cols.remove(j)
                removed.append({'player':'col','action':j,'dominated_by':winner,
                                'opponent_actions':rows.copy()})
                change=True
                break
        if not change:
            return rows,cols,removed


def _zero_sum_candidate(a,b):
    """Primal/dual minimax LP for a constant-sum game; checked after solving."""
    aa=np.asarray(a,dtype=np.longdouble)
    bb=np.asarray(b,dtype=np.longdouble)
    combined=aa+bb
    if not np.all(combined==combined.flat[0]):
        return None
    lo=aa.min();hi=aa.max()
    center=lo+(hi-lo)/2
    spread=np.max(np.abs(aa-center))
    if spread==0:
        spread=np.longdouble(1)
    matrix=np.asarray((aa-center)/spread,dtype=float)
    m,n=matrix.shape
    p=linprog(np.r_[np.zeros(m),-1.],A_ub=np.c_[-matrix.T,np.ones(n)],
              b_ub=np.zeros(n),A_eq=np.array([[*([1.]*m),0.]]),
              b_eq=[1.],bounds=[(0,None)]*m+[(None,None)],method='highs')
    q=linprog(np.r_[np.zeros(n),1.],A_ub=np.c_[matrix,-np.ones(m)],
              b_ub=np.zeros(m),A_eq=np.array([[*([1.]*n),0.]]),
              b_eq=[1.],bounds=[(0,None)]*n+[(None,None)],method='highs')
    if not (p.success and q.success):
        return None
    x=np.maximum(p.x[:m],0)
    y=np.maximum(q.x[:n],0)
    if x.sum()==0 or y.sum()==0:
        return None
    x=x/x.sum();y=y/y.sum()
    normalized_gap=max(0.,float(np.max(matrix@y)-np.min(x@matrix)))
    if normalized_gap > 1e-7:
        return None
    return x,y,{'normalized_minimax_duality_gap':normalized_gap,
                'constant_sum': True}


def solve_general_nash(a,b,*,mode='auto',iterations=1000,restarts=2,seed=0,
                       max_exact_pairs=96,max_exact_operations=20000,
                       max_exact_support=3, max_pure_cells=100000):
    """Size-independent algorithm portfolio; API may enforce separate limits.

    mode='auto': pure, constant-sum, bounded LH, then approximate.
    mode='approximate': only the stochastic best-response baseline.
    mode='exact': budgeted exact rational cell search (possibly partial).
    """
    a,b=_validate(a,b)
    if mode not in ('auto','approximate','exact'):
        raise ValueError('Invalid general solver mode')
    if (type(max_pure_cells) is not int or not 1<=max_pure_cells<=1000000 or
            type(iterations) is not int or not 100<=iterations<=200000 or
            type(restarts) is not int or not 1<=restarts<=64 or
            iterations*restarts>500000 or
            type(seed) is not int or not 0<=seed<=2147483647):
        raise ValueError('Invalid general solver budget')
    rows,cols,eliminations=eliminate_strict_pure_dominance(a,b)
    reduced_a=a[np.ix_(rows,cols)]
    reduced_b=b[np.ix_(rows,cols)]
    candidates=[]
    def add(x,y,method,kind,additional=None):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
        if (x.shape!=(len(rows),) or y.shape!=(len(cols),) or
                not np.isfinite(x).all() or not np.isfinite(y).all() or
                (x<0).any() or (y<0).any()):
            return False
        full_x=np.zeros(a.shape[0]);full_y=np.zeros(a.shape[1])
        full_x[rows]=x;full_y[cols]=y
        certificate=certify_bimatrix(a,b,full_x,full_y)
        result={'row_probabilities':full_x.tolist(),
                'col_probabilities':full_y.tolist(),
                'method':method,'evidence':kind,
                'certificate':certificate}
        if additional:
            result.update(additional)
        candidates.append(result)
        return True
    pure_examined=False
    if mode=='auto' and len(rows)*len(cols)<=max_pure_cells:
        pure_examined=True
        for i in range(len(rows)):
            for j in range(len(cols)):
                if (reduced_a[i,j]==np.max(reduced_a[:,j]) and
                        reduced_b[i,j]==np.max(reduced_b[i,:])):
                    x=np.eye(len(rows))[i]
                    y=np.eye(len(cols))[j]
                    add(x,y,'pure_best_response','exact_pure_equilibrium')
                    break
            if candidates:
                break
    exact=None
    if mode=='exact':
        exact=search_exact_nash_cells(reduced_a,reduced_b,max_pairs=max_exact_pairs,
                        max_operations=max_exact_operations,max_support=max_exact_support)
        for family in exact['families'][:1]:
            x=[float(Fraction(p)) for p in family['row_probabilities_exact']]
            y=[float(Fraction(p)) for p in family['col_probabilities_exact']]
            add(x,y,'rational_support_witness','exact_rational_witness',
                {'row_probabilities_exact':family['row_probabilities_exact'],
                 'col_probabilities_exact':family['col_probabilities_exact']})
    if not candidates and mode=='auto':
        zs=_zero_sum_candidate(reduced_a,reduced_b)
        if zs is not None:
            add(zs[0],zs[1],'constant_sum_minimax_lp','numerically_checked',zs[2])
        if not candidates and len(rows)>1 and len(cols)>1 and len(rows)+len(cols)<=10:
            lh=solve_independent_lh(reduced_a,reduced_b,max_labels=min(4,len(rows)+len(cols)),
                                    max_pivots=250)
            for equilibrium in lh['equilibria'][:1]:
                add(equilibrium['row_probabilities'],equilibrium['col_probabilities'],
                    'independent_lemke_howson','numerically_checked')
    if not candidates and mode in ('auto','approximate'):
        outcome=approximate_nash(reduced_a,reduced_b,iterations=iterations,
                                 restarts=restarts,seed=seed)
        profile=outcome['profile']
        if profile is not None:
            add(profile['row_probabilities'],profile['col_probabilities'],
                'multi_start_fictitious_play','approximate_candidate',
                {'iterations_used':profile['iteration'],'seed':seed})
    if exact is not None:
        exact['strategy_coordinates']='reduced game (index maps in retained_actions)'
    verified=any(c['certificate']['valid'] for c in candidates)
    return {'method':'general_nash_v45',
            'original_shape':list(a.shape),
            'reduced_shape':list(reduced_a.shape),
            'retained_actions':{'row':rows,'col':cols},
            'strict_dominance_eliminations':eliminations,
            'pure_search_exhausted':pure_examined,
            'candidates':candidates,
            'best_candidate':next((c for c in candidates if c['certificate']['valid']),
                                  candidates[0] if candidates else None),
            'candidate_certified_within_tolerance':verified,
            'equilibrium_set_complete':bool(exact and exact['complete_equilibrium_set']),
            'exact_search':exact,
            'status':('complete_exact_set' if exact and exact['complete_equilibrium_set']
                      else 'partial_exact_search' if exact
                      else 'certified_candidate' if verified
                      else 'uncertified_candidate' if candidates else 'inconclusive'),
            'note':('A numerical deviation certificate applies to one candidate, not all equilibria. '
                    'Complete-set claims require exhaustion of every support with exact rational feasibility.'),
            }
