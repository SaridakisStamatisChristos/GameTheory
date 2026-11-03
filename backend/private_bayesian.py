"""Finite private-type Bayesian games: exact pure contingent-policy equilibrium checks."""
from itertools import product
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

class PrivateBayesianGame(BaseModel):
    joint_prior: list[list[float]]
    row_payoffs: list[list[list[list[float]]]]
    col_payoffs: list[list[list[list[float]]]]
    row_policy: list[int]
    col_policy: list[int]

def evaluate_private(game: PrivateBayesianGame):
    try:
        p=np.asarray(game.joint_prior,dtype=float)
        a=np.asarray(game.row_payoffs,dtype=float)
        b=np.asarray(game.col_payoffs,dtype=float)
    except (ValueError,TypeError):
        raise HTTPException(422,'Nonrectangular or nonnumeric input')
    if (p.ndim!=2 or not p.size or max(p.shape)>12 or a.ndim!=4 or a.shape!=b.shape
        or a.shape[:2]!=p.shape or not a.shape[2] or not a.shape[3]
        or max(a.shape[2:])>12 or not all(np.isfinite(z).all() for z in (p,a,b))
        or (p<0).any() or abs(p.sum()-1)>1e-8):
        raise HTTPException(422,'Invalid private Bayesian game dimensions or prior')
    t,u=p.shape;m,n=a.shape[2:]
    if len(game.row_policy)!=t or len(game.col_policy)!=u or any(type(x)!=int or x<0 or x>=m for x in game.row_policy) or any(type(x)!=int or x<0 or x>=n for x in game.col_policy):
        raise HTTPException(422,'Policy must specify a valid pure action for each private type')
    row=np.asarray(game.row_policy);col=np.asarray(game.col_policy)
    from safe_numeric import json_number
    P=np.asarray(p,dtype=np.longdouble)
    A=np.asarray(a,dtype=np.longdouble); B=np.asarray(b,dtype=np.longdouble)
    expected=np.zeros(2,dtype=np.longdouble);row_gains=[];col_gains=[];row_cond=[];col_cond=[];all_gains=[]
    for i in range(t):
        mass=np.sum(P[i,:])
        pay=np.array([sum(P[i,j]*A[i,j,k,col[j]] for j in range(u)) for k in range(m)],dtype=np.longdouble)
        gain=max(np.longdouble(0),np.max(pay)-pay[row[i]])
        all_gains.append(gain)
        row_gains.append(json_number(gain))
        row_cond.append(json_number(gain/mass) if mass>0 else None)
    for j in range(u):
        mass=np.sum(P[:,j])
        pay=np.array([sum(P[i,j]*B[i,j,row[i],k] for i in range(t)) for k in range(n)],dtype=np.longdouble)
        gain=max(np.longdouble(0),np.max(pay)-pay[col[j]])
        all_gains.append(gain)
        col_gains.append(json_number(gain))
        col_cond.append(json_number(gain/mass) if mass>0 else None)
    for i,j in product(range(t),range(u)):
        expected+=P[i,j]*np.array([A[i,j,row[i],col[j]],B[i,j,row[i],col[j]]])
    exact_best_responses=all(g==0 for g in all_gains)
    return {'expected_payoffs':[json_number(z) for z in expected],
            'row_type_ex_ante_deviation_gains':row_gains,'col_type_ex_ante_deviation_gains':col_gains,
            'row_type_conditional_deviation_gains':row_cond,'col_type_conditional_deviation_gains':col_cond,
            'is_bayesian_nash':exact_best_responses,
            'exact_payoff_comparison':True,
            'note':'BNE checks positive-probability types for conditional profitable deviations; ex-ante gains are separate diagnostics.'}
