from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api_guard import LabGuard
from pydantic import BaseModel, Field
from typing import List, Literal
import numpy as np
from scipy.optimize import linprog

app = FastAPI(title='Game Theory Lab API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173'], allow_methods=['*'], allow_headers=['*'])
app.add_middleware(LabGuard)

class Game(BaseModel):
    row_payoffs: List[List[float]] = Field(min_length=1)
    col_payoffs: List[List[float]] = Field(min_length=1)

def validate(game):
    try:
        a = np.asarray(game.row_payoffs, dtype=float)
        b = np.asarray(game.col_payoffs, dtype=float)
    except (ValueError, TypeError, OverflowError):
        raise HTTPException(422, "Payoffs must be rectangular numeric matrices")
    if a.ndim != 2 or b.shape != a.shape or not a.size or a.shape[0] > 20 or a.shape[1] > 20 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise HTTPException(422, 'Payoffs must be finite, rectangular, same shape, at most 20x20')
    return a,b

@app.get('/api/health')
def health(): return {'status':'ok'}

@app.post('/api/analyze')
def analyze(game:Game):
    a,b=validate(game)
    m,n=a.shape
    pure=[]
    for i in range(m):
        for j in range(n):
            if a[i,j] >= np.max(a[:,j]) and b[i,j] >= np.max(b[i,:]):
                pure.append({'row':i,'col':j,'payoffs':[float(a[i,j]),float(b[i,j])]})
    row_dominated=[]; col_dominated=[]
    for i in range(m):
        for k in range(m):
            if i!=k and np.all(a[k,:]>a[i,:]):
                row_dominated.append({'strategy':i,'by':k});break
    for j in range(n):
        for k in range(n):
            if j!=k and np.all(b[:,k]>b[:,j]):
                col_dominated.append({'strategy':j,'by':k});break
    mixed=[]
    if (m,n)==(2,2):
        # Interior mixed equilibrium: row player indifferent between rows, column between columns.
        from fractions import Fraction
        from safe_numeric import expected_payoffs
        # Rational indifference avoids an arbitrary absolute nondegeneracy cutoff.
        af = [[Fraction(str(v)) for v in row] for row in a]
        bf = [[Fraction(str(v)) for v in row] for row in b]
        d=af[0][0]-af[0][1]-af[1][0]+af[1][1]
        e=bf[0][0]-bf[1][0]-bf[0][1]+bf[1][1]
        if d and e:
            q=(af[1][1]-af[0][1])/d
            p=(bf[1][1]-bf[1][0])/e
            if 0<p<1 and 0<q<1:
                x=np.array([float(p),float(1-p)]);y=np.array([float(q),float(1-q)])
                mixed.append({'row_probabilities':x.tolist(),'col_probabilities':y.tolist(),
                              'expected_payoffs':expected_payoffs(a,b,x,y),
                              'certificate':'exact rational indifference on input decimal payoffs'})
    return {'shape':[m,n],'pure_nash':pure,'strictly_dominated_by_pure':{'row':row_dominated,'col':col_dominated},'interior_mixed_nash_2x2':mixed,'note':'Mixed equilibrium calculation is limited to nondegenerate interior 2x2 cases; pure dominance uses pure strategies only.'}

class ZeroSum(BaseModel):
    payoffs:List[List[float]]

@app.post('/api/zero-sum')
def zero_sum(data:ZeroSum):
    try:
        a=np.asarray(data.payoffs,dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(422, "Invalid rectangular payoff matrix") from exc
    if a.ndim!=2 or not a.size or max(a.shape)>20 or not np.isfinite(a).all():
        raise HTTPException(422,'Invalid matrix')
    from safe_numeric import json_number
    aa=np.asarray(a,dtype=np.longdouble)
    # Positive affine transforms preserve minimax strategies. Center before
    # converting the coefficients to float64 to avoid HiGHS tolerance collapse.
    lo=np.min(aa); hi=np.max(aa); center=lo+(hi-lo)/2
    spread=np.max(np.abs(aa-center))
    if spread==0: spread=np.longdouble(1)
    normalized=np.asarray((aa-center)/spread,dtype=float)
    m,n=normalized.shape
    c=np.r_[np.zeros(m),-1.0]
    res=linprog(c,A_ub=np.c_[-normalized.T,np.ones(n)],b_ub=np.zeros(n),
                A_eq=np.array([[*([1.0]*m),0.0]]),b_eq=[1],
                bounds=[(0,None)]*m+[(None,None)],method='highs')
    res2=linprog(np.r_[np.zeros(n),1.0],
                 A_ub=np.c_[normalized,-np.ones(m)],b_ub=np.zeros(m),
                 A_eq=np.array([[*([1.0]*n),0.0]]),b_eq=[1],
                 bounds=[(0,None)]*n+[(None,None)],method='highs')
    if not res.success or not res2.success:
        raise HTTPException(422,'Minimax solver was numerically inconclusive')
    x=np.maximum(res.x[:m],0);y=np.maximum(res2.x[:n],0)
    x/=x.sum();y/=y.sum()
    # Verify in the same well-conditioned coordinate system as the LP.
    lower=float(np.min(x@normalized));upper=float(np.max(normalized@y))
    if upper-lower > 1e-7:
        raise HTTPException(422,'Minimax certificate failed in normalized units')
    value=np.asarray(x,dtype=np.longdouble)@aa@np.asarray(y,dtype=np.longdouble)
    return {'row_strategy':x.tolist(),'col_strategy':y.tolist(),
            'value':json_number(value),
            'duality_gap':json_number(np.longdouble(max(0.,upper-lower))*spread),
            'normalized_duality_gap':max(0.,upper-lower),
            'certificate_units':'normalized payoff span and original payoff units'}

from advanced import support_equilibria, replicator

@app.post('/api/equilibria')
def equilibria(game: Game):
    import math
    a,b=validate(game)
    m,n=a.shape
    work=sum(math.comb(m,k)*math.comb(n,k) for k in range(1,min(4,m,n)+1))
    if work>10_000:raise HTTPException(422,'Fast support candidate budget exceeded')
    result=support_equilibria(a,b)
    return {'equilibria':result,'complete':False,'note':'Nondegenerate equal-size support enumeration (up to size 4). Degenerate and unequal-support equilibria may be missed.'}

class EvolutionRequest(BaseModel):
    payoffs:List[List[float]]
    initial:List[float]|None=None
    steps:int=Field(default=200,ge=1,le=2000)
    dt:float=Field(default=0.05,gt=0,le=0.25)

@app.post('/api/evolution')
def evolution(req:EvolutionRequest):
    return replicator(req.payoffs,req.initial,req.steps,req.dt)

from extended import TreeGame, RepeatedGame, backward_induction, repeated_payoffs

@app.post('/api/tree/backward-induction')
def tree_backward_induction(game: TreeGame):
    return backward_induction(game)

@app.post('/api/repeated/simulate')
def repeated_simulate(game: RepeatedGame):
    return repeated_payoffs(game)

from research import BayesianGame, Experiment, bayesian_evaluate, correlated_equilibrium, save_experiment, list_experiments, get_experiment

@app.post('/api/correlated-equilibrium')
def correlated(game: Game):
    a,b=validate(game)
    return correlated_equilibrium(a,b)

from exact_correlated import solve_exact_correlated_equilibrium

class ExactCorrelatedRequest(Game):
    objective: Literal['welfare','feasible'] = 'welfare'
    max_operations: int = Field(default=200000, ge=1, le=2000000)

@app.post('/api/correlated-equilibrium/exact')
def correlated_exact(req: ExactCorrelatedRequest):
    a,b=validate(req)
    try:
        return solve_exact_correlated_equilibrium(
            a,b,objective=req.objective,max_operations=req.max_operations)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc

@app.post('/api/bayesian/evaluate')
def bayesian(game: BayesianGame):
    return bayesian_evaluate(game)

@app.post('/api/experiments')
def create_experiment(exp: Experiment):
    return save_experiment(exp)

@app.get('/api/experiments')
def experiments():
    return list_experiments()

@app.get('/api/experiments/{experiment_id}')
def experiment(experiment_id: str):
    return get_experiment(experiment_id)

from private_bayesian import PrivateBayesianGame, evaluate_private

@app.post('/api/bayesian/private-evaluate')
def private_bayesian(game: PrivateBayesianGame):
    return evaluate_private(game)

from approximate import approximate_nash

class ApproximateNashRequest(Game):
    iterations: int = Field(default=10000, ge=100, le=200000)
    restarts: int = Field(default=8, ge=1, le=64)
    seed: int = Field(default=0, ge=0, le=2147483647)

@app.post('/api/equilibria/approximate')
def approximate_equilibria(game: ApproximateNashRequest):
    a, b = validate(game)
    return approximate_nash(a, b, game.iterations, game.restarts, game.seed)

from cfr import CFRRequest, solve_cfr

@app.post('/api/imperfect-information/cfr')
def imperfect_information_cfr(request: CFRRequest):
    return solve_cfr(request)

from cfr_plus import CFRPlusRequest, solve_cfr_plus

@app.post('/api/imperfect-information/cfr-plus')
def imperfect_information_cfr_plus(request: CFRPlusRequest):
    return solve_cfr_plus(request)

from support_lp import solve_support_lp

class SupportLPRequest(Game):
    max_pairs: int = Field(default=100000, ge=1, le=1000000)

@app.post('/api/equilibria/support-lp')
def equilibria_support_lp(req: SupportLPRequest):
    a,b=validate(req)
    if (2**a.shape[0]-1)*(2**a.shape[1]-1)>5_000:
        raise HTTPException(422,'Service support-pair work budget exceeded (5000)')
    try:
        return solve_support_lp(a,b,max_pairs=req.max_pairs)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc

from hybrid_nash import solve_hybrid

class HybridNashRequest(Game):
    max_pairs: int = Field(default=100000, ge=1, le=1000000)
    fast_support: int = Field(default=4, ge=1, le=20)
    exhaustive: bool = False

@app.post('/api/equilibria/hybrid')
def equilibria_hybrid(req: HybridNashRequest):
    a,b=validate(req)
    import math
    m,n=a.shape
    work=sum(math.comb(m,k)*math.comb(n,k) for k in range(1,min(req.fast_support,m,n)+1))
    if work>10_000: raise HTTPException(422,'Hybrid fast-support candidate budget exceeded (10000)')
    if req.exhaustive and (2**m-1)*(2**n-1)>5_000:
        raise HTTPException(422,'Hybrid exhaustive support-pair budget exceeded (5000)')
    try:
        return solve_hybrid(a,b,max_pairs=min(req.max_pairs,5_000),fast_support=req.fast_support,exhaustive=req.exhaustive)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc

from lemke_howson import solve_lemke_howson

class LemkeHowsonRequest(Game):
    max_labels: int | None = Field(default=None, ge=1, le=40)

@app.post('/api/equilibria/lemke-howson')
def equilibria_lemke_howson(req: LemkeHowsonRequest):
    a,b=validate(req)
    try:
        return solve_lemke_howson(a,b,max_labels=req.max_labels)
    except (ValueError,RuntimeError) as exc:
        raise HTTPException(422,str(exc)) from exc

from analytic_2x2 import solve_2x2

@app.post('/api/equilibria/analytic-2x2')
def analytic_2x2_endpoint(game: Game):
    a,b=validate(game)
    if a.shape != (2,2):
        raise HTTPException(422,'Analytic solver requires 2x2 matrices')
    return solve_2x2(a,b)

from independent_lh import solve_independent_lh

@app.post('/api/equilibria/independent-lemke-howson')
def equilibria_independent_lemke_howson(req: LemkeHowsonRequest):
    a, b = validate(req)
    try:
        return solve_independent_lh(a, b, max_labels=req.max_labels)
    except (ValueError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

from exact_support import solve_exact_support

class ExactSupportRequest(Game):
    max_pairs: int = Field(default=225, ge=1, le=961)
    max_operations: int = Field(default=100000, ge=1, le=2000000)
    max_actions: int = Field(default=4, ge=4, le=5)

@app.post('/api/equilibria/exact-support')
def equilibria_exact_support(req: ExactSupportRequest):
    a,b=validate(req)
    if max(a.shape)>req.max_actions:
        raise HTTPException(422,'Exact support feasibility exceeds requested action dimension')
    if req.max_actions==4 and (req.max_pairs>225 or req.max_operations>500000):
        raise HTTPException(422,'Default 4x4 budget ceilings exceeded')
    try:
        return solve_exact_support(a,b,max_pairs=req.max_pairs,max_operations=req.max_operations,
                                   max_actions=req.max_actions)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc

from exact_nash_set import solve_exact_nash_set

class ExactNashSetRequest(Game):
    max_pairs: int = Field(default=225, ge=1, le=961)
    max_operations: int = Field(default=100000, ge=1, le=2000000)
    max_actions: int = Field(default=4, ge=4, le=5)
    include_vertices: bool = False
    max_vertex_candidates: int = Field(default=50000, ge=1, le=200000)

@app.post('/api/equilibria/exact-nash-set')
def equilibria_exact_nash_set(req: ExactNashSetRequest):
    a,b=validate(req)
    if max(a.shape)>req.max_actions:
        raise HTTPException(422,'Exact Nash geometry exceeds requested action dimension')
    if req.max_actions==4 and (req.max_pairs>225 or req.max_operations>500000):
        raise HTTPException(422,'Default 4x4 budget ceilings exceeded')
    try:
        return solve_exact_nash_set(a,b,max_pairs=req.max_pairs,max_operations=req.max_operations,
                                    max_actions=req.max_actions,include_vertices=req.include_vertices,
                                    max_vertex_candidates=req.max_vertex_candidates)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc

from general_nash import solve_general_nash

class GeneralNashRequest(Game):
    mode: Literal['auto','approximate','exact'] = 'auto'
    iterations: int = Field(default=1000,ge=100,le=200000)
    restarts: int = Field(default=2,ge=1,le=64)
    seed: int = Field(default=0,ge=0,le=2147483647)
    max_pure_cells: int = Field(default=100000,ge=1,le=1000000)
    max_exact_pairs: int = Field(default=96,ge=1,le=4096)
    max_exact_operations: int = Field(default=20000,ge=1,le=2000000)
    max_exact_support: int = Field(default=3,ge=1,le=20)
    exact_backend: Literal['simplex','vertices'] = 'simplex'
    prune_supports: bool = True
    mixed_pruning: bool = False

@app.post('/api/equilibria/general')
def equilibria_general(req: GeneralNashRequest):
    a,b=validate(req)
    try:
        return solve_general_nash(a,b,mode=req.mode,iterations=req.iterations,
            restarts=req.restarts,seed=req.seed,max_pure_cells=req.max_pure_cells,
            max_exact_pairs=req.max_exact_pairs,
            max_exact_operations=req.max_exact_operations,
            max_exact_support=req.max_exact_support,exact_backend=req.exact_backend,
            prune_supports=req.prune_supports,mixed_pruning=req.mixed_pruning)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
