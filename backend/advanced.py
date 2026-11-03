import itertools
import numpy as np
from scipy.optimize import linprog
from fastapi import HTTPException

def support_equilibria(a, b, tol=1e-7, max_support=4):
    """Enumerate nondegenerate equilibria with equal-size supports, verifying all deviations.
    Not complete for degenerate games or supports larger than max_support.
    """
    from safe_numeric import expected_payoffs
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.ndim!=2 or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected nonempty finite bimatrices')
    m,n=a.shape
    def scaled(matrix):
        mm=np.asarray(matrix,dtype=np.longdouble)
        low=np.min(mm);r=np.max(mm)-low
        return np.asarray((mm-low)/(r if r else np.longdouble(1)),dtype=float)
    aa=scaled(a);bb=scaled(b)
    found=[]
    for k in range(1,min(m,n,max_support)+1):
        for I in itertools.combinations(range(m),k):
            for J in itertools.combinations(range(n),k):
                A=np.block([[aa[np.ix_(I,J)],-np.ones((k,1))],[np.ones((1,k)),np.zeros((1,1))]])
                B=np.block([[bb[np.ix_(I,J)].T,-np.ones((k,1))],[np.ones((1,k)),np.zeros((1,1))]])
                rhs=np.r_[np.zeros(k),1.]
                try:
                    y0=np.linalg.solve(A,rhs)[:k]
                    x0=np.linalg.solve(B,rhs)[:k]
                except np.linalg.LinAlgError:
                    continue
                if min(x0.min(),y0.min())<=0:continue
                x=np.zeros(m);y=np.zeros(n);x[list(I)]=x0;y[list(J)]=y0
                if k==1:
                    # Pure best responses are exactly comparable on the inputs.
                    if a[I[0],J[0]]<np.max(a[:,J[0]]) or b[I[0],J[0]]<np.max(b[I[0],:]):continue
                elif np.max(aa@y)-x@aa@y>1e-9 or np.max(x@bb)-x@bb@y>1e-9:
                    continue
                if any(np.max(np.abs(x-np.array(z['row_probabilities'])))<tol and
                       np.max(np.abs(y-np.array(z['col_probabilities'])))<tol for z in found):continue
                found.append({'row_probabilities':x.tolist(),'col_probabilities':y.tolist(),
                              'expected_payoffs':expected_payoffs(a,b,x,y),
                              'support':{'row':list(I),'col':list(J)},
                              'verification':'numerical, player-normalized' if k>1 else 'pure best response exact'})
    return found

def replicator(payoffs, initial=None, steps=200, dt=0.05):
    try:
        a=np.asarray(payoffs,dtype=float)
    except (ValueError,TypeError,OverflowError) as exc:
        raise HTTPException(422,'Expected finite square payoff matrix') from exc
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or a.shape[0]>20 or not np.isfinite(a).all():
        raise HTTPException(422,'Expected finite square payoff matrix up to 20x20')
    n=a.shape[0]
    try:
        x=np.asarray(initial if initial is not None else [1/n]*n,dtype=np.longdouble)
    except (ValueError,TypeError,OverflowError) as exc:
        raise HTTPException(422,'Invalid initial population') from exc
    if x.shape!=(n,) or not np.isfinite(x).all() or (x<0).any() or x.sum()<=0:
        raise HTTPException(422,'Invalid initial population')
    if not 1<=steps<=2000 or not 0<dt<=0.25:raise HTTPException(422,'Invalid integration parameters')
    x=np.asarray(x/x.sum(),dtype=float);trajectory=[x.tolist()]
    for _ in range(steps):
        # Multiplicative-weights discretization: preserves simplex and positivity.
        fitness=np.asarray(a,dtype=np.longdouble)@np.asarray(x,dtype=np.longdouble)
        active=x>0
        logits=np.log(np.asarray(x[active],dtype=np.longdouble))+np.longdouble(dt)*(fitness[active]-np.max(fitness[active]))
        weights=np.exp(logits-np.max(logits))
        nxt=np.zeros_like(x)
        nxt[active]=np.asarray(weights/np.sum(weights),dtype=float)
        x=nxt / np.sum(nxt)
        trajectory.append(x.tolist())
    return {'trajectory':trajectory,'final_distribution':x.tolist(),'method':'exponential replicator discretization','dt':dt}
