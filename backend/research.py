"""Verified finite-game research primitives."""
import sqlite3, json, os, uuid
from contextlib import closing
import numpy as np
from scipy.optimize import linprog
from fastapi import HTTPException
from pydantic import BaseModel, Field

class BayesianGame(BaseModel):
    prior: list[float]
    row_payoffs: list[list[list[float]]]
    col_payoffs: list[list[list[float]]]
    row_strategy: list[int]
    col_strategy: list[int]

def bayesian_evaluate(req: BayesianGame):
    """Common-prior finite state model with state-contingent actions; reports ex-ante payoff and statewise deviations."""
    p=np.asarray(req.prior,dtype=float)
    a=np.asarray(req.row_payoffs,dtype=float); b=np.asarray(req.col_payoffs,dtype=float)
    if p.ndim!=1 or not len(p) or len(p)>32 or not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>1e-8:
        raise HTTPException(422,'Prior must be a finite probability distribution summing to one')
    if a.ndim!=3 or a.shape!=b.shape or a.shape[0]!=len(p) or not np.isfinite(a).all() or not np.isfinite(b).all() or max(a.shape[1:])>20:
        raise HTTPException(422,'Invalid state-dependent bimatrices')
    if len(req.row_strategy)!=len(p) or len(req.col_strategy)!=len(p):
        raise HTTPException(422,'One action per state required')
    vals=[]; regrets=[]
    for s,(i,j) in enumerate(zip(req.row_strategy,req.col_strategy)):
        if not 0<=i<a.shape[1] or not 0<=j<a.shape[2]:raise HTTPException(422,'Action out of range')
        vals.append([float(a[s,i,j]),float(b[s,i,j])])
        regrets.append([float(max(0,np.max(a[s,:,j])-a[s,i,j])),float(max(0,np.max(b[s,i,:])-b[s,i,j]))])
    return {'expected_payoffs':(p@np.asarray(vals)).tolist(),'state_payoffs':vals,'state_deviation_gains':regrets,'is_statewise_nash':bool(np.max(regrets)==0),'assumption':'Both players observe the state before acting; this is not a private-type Bayesian Nash solver'}

def correlated_equilibrium(a,b,objective='welfare'):
    """LP with per-player conditioning and independent original-unit obedience check.

    With disparate utility scales, constraints MUST NOT share a common scale:
    otherwise meaningful unilateral gains disappear below solver tolerance.
    'welfare' maximizes the sum of *per-player normalized* expected utilities;
    it is not necessarily the raw-payoff utilitarian optimum.
    """
    from safe_numeric import json_number
    a=np.asarray(a,dtype=np.longdouble);b=np.asarray(b,dtype=np.longdouble)
    m,n=a.shape;dim=m*n
    def condition(payoff):
        lo=np.min(payoff); spread=np.max(payoff)-lo
        if spread==0: spread=np.longdouble(1)
        return np.asarray((payoff-lo)/spread,dtype=float)
    aa=condition(a);bb=condition(b)
    rows=[]
    for i in range(m):
        for k in range(m):
            if k!=i:
                v=np.zeros((m,n));v[i,:]=aa[k,:]-aa[i,:];rows.append(v.ravel())
    for j in range(n):
        for k in range(n):
            if k!=j:
                v=np.zeros((m,n));v[:,j]=bb[:,k]-bb[:,j];rows.append(v.ravel())
    if objective not in ('welfare','feasible'):
        raise HTTPException(422,'Unsupported correlated-equilibrium objective')
    c=-(aa+bb).ravel() if objective=='welfare' else np.zeros(dim)
    result=linprog(c,A_ub=np.asarray(rows) if rows else None,
                   b_ub=np.zeros(len(rows)) if rows else None,
                   A_eq=np.ones((1,dim)),b_eq=[1.],bounds=(0,None),method='highs')
    if not result.success:raise HTTPException(422,'Correlated-equilibrium LP was inconclusive')
    q=np.maximum(result.x.reshape((m,n)),0);q=q/q.sum()
    ql=np.asarray(q,dtype=np.longdouble)
    # Validate obedience from original payoffs, independently of the LP matrix.
    gains=[];normalized_gains=[]
    for matrix in (a,b):
        scale=np.max(matrix)-np.min(matrix)
        if scale==0: scale=np.longdouble(1)
        raw=[]
        if matrix is a:
            for i in range(m):
                for k in range(m):
                    if i!=k:raw.append(np.sum(ql[i,:]*(matrix[k,:]-matrix[i,:])))
        else:
            for j in range(n):
                for k in range(n):
                    if j!=k:raw.append(np.sum(ql[:,j]*(matrix[:,k]-matrix[:,j])))
        gap=max(np.longdouble(0),*raw)
        gains.append(gap);normalized_gains.append(gap/scale)
    # No categorical mathematical claim from a badly conditioned LP solution.
    if max(normalized_gains)>np.longdouble('1e-9') or max(gains)>np.longdouble('1e-7'):
        raise HTTPException(422,'Correlated-equilibrium original-unit obedience certificate failed')
    return {'joint_distribution':q.tolist(),
            'expected_payoffs':[json_number(np.sum(ql*matrix)) for matrix in (a,b)],
            'maximum_obedience_violation':json_number(max(gains)),
            'maximum_normalized_obedience_violation':float(max(normalized_gains)),
            'per_player_obedience_violation':[json_number(g) for g in gains],
            'objective':objective,'objective_units':'sum of individual normalized player payoffs',
            'solution':'numerically certified correlated-equilibrium candidate',
            'certificate_status':'original-unit and per-player-normalized obedience checks passed within declared tolerances',
            'violation_units':'original payoff units'}

class Experiment(BaseModel):
    title: str = Field(min_length=1,max_length=160)
    game: dict
    notes: str = Field(default='',max_length=5000)

def db_connection():
    path=os.getenv('GAME_LAB_DB','game_lab.sqlite3')
    con=sqlite3.connect(path,timeout=5)
    con.execute('CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, title TEXT NOT NULL, game TEXT NOT NULL, notes TEXT NOT NULL)')
    return con

def save_experiment(exp:Experiment):
    ident=str(uuid.uuid4())
    try:
        encoded=json.dumps(exp.game,allow_nan=False)
    except (TypeError,ValueError,OverflowError) as exc:
        raise HTTPException(422,'Experiment must be finite and JSON serializable') from exc
    if len(encoded.encode('utf-8'))>200_000:
        raise HTTPException(413,'Experiment exceeds storage size budget')
    with closing(db_connection()) as db:
        with db:
            db.execute('BEGIN IMMEDIATE')  # serialize quota check with inserts
            count,size=db.execute('SELECT COUNT(*), COALESCE(SUM(LENGTH(game)+LENGTH(notes)+LENGTH(title)),0) FROM experiments').fetchone()
            if count>=1000 or size+len(encoded)+len(exp.notes)+len(exp.title)>10_000_000:
                raise HTTPException(507,'Local experiment storage quota exceeded')
            db.execute('INSERT INTO experiments VALUES (?,?,?,?)',(ident,exp.title,encoded,exp.notes))
    return {'id':ident,'title':exp.title}

def list_experiments():
    with closing(db_connection()) as db:rows=db.execute('SELECT id,title FROM experiments ORDER BY rowid DESC LIMIT 100').fetchall()
    return [{'id':r[0],'title':r[1]} for r in rows]

def get_experiment(ident:str):
    with closing(db_connection()) as db:row=db.execute('SELECT id,title,game,notes FROM experiments WHERE id=?',(ident,)).fetchone()
    if row is None:raise HTTPException(404,'Experiment not found')
    return {'id':row[0],'title':row[1],'game':json.loads(row[2]),'notes':row[3]}
