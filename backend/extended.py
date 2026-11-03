"""Finite perfect-information extensive games and repeated-game analysis."""
from functools import lru_cache
import math
from fastapi import HTTPException
from pydantic import BaseModel, Field

class TreeNode(BaseModel):
    id: str
    player: int | None = None
    children: list[str] = Field(default_factory=list)
    payoffs: list[float] | None = None

class TreeGame(BaseModel):
    root: str
    nodes: list[TreeNode] = Field(min_length=1, max_length=500)
    players: int = Field(default=2, ge=2, le=8)

def backward_induction(game: TreeGame):
    nodes = {node.id:node for node in game.nodes}
    if len(nodes)!=len(game.nodes) or game.root not in nodes:
        raise HTTPException(422,'Duplicate node IDs or unknown root')
    for node in game.nodes:
        if node.payoffs is not None:
            if node.children or node.player is not None or len(node.payoffs)!=game.players or not all(math.isfinite(v) for v in node.payoffs):
                raise HTTPException(422,'Invalid terminal node')
        elif node.player is None or not 0<=node.player<game.players or not node.children:
            raise HTTPException(422,'Nonterminal node requires player and children')
        if any(child not in nodes for child in node.children):
            raise HTTPException(422,'Unknown child node')
    state={}; policy={}; visited=set()
    def solve(node_id):
        if state.get(node_id)==1: raise HTTPException(422,'Cycle detected')
        if state.get(node_id)==2: return visited_values[node_id]
        state[node_id]=1;visited.add(node_id)
        node=nodes[node_id]
        if node.payoffs is not None: value=node.payoffs
        else:
            choices=[(child,solve(child)) for child in node.children]
            best=max(v[node.player] for _,v in choices)
            winners=[child for child,v in choices if v[node.player]==best]
            chosen=next(child for child,_ in choices if child in winners)
            policy[node_id]={'chosen':chosen,'optimal_children':winners,'player':node.player}
            value=next(v for child,v in choices if child==chosen)
        state[node_id]=2;visited_values[node_id]=value
        return value
    visited_values={}
    outcome=solve(game.root)
    if len(visited)!=len(nodes):raise HTTPException(422,'Unreachable nodes are not allowed')
    path=[game.root]
    while path[-1] in policy:path.append(policy[path[-1]]['chosen'])
    return {'payoffs':outcome,'path':path,'policy':policy,'tie_break':'first listed optimal child','solution':'subgame-perfect equilibrium by backward induction (one pure selection)'}

class RepeatedGame(BaseModel):
    row_payoffs: list[list[float]]
    col_payoffs: list[list[float]]
    row_strategy: list[int]
    col_strategy: list[int]
    rounds: int = Field(default=10, ge=1, le=10000)
    discount: float = Field(default=1.0, ge=0, le=1)

def repeated_payoffs(req:RepeatedGame):
    from main import validate, Game
    a,b=validate(Game(row_payoffs=req.row_payoffs,col_payoffs=req.col_payoffs))
    if not req.row_strategy or not req.col_strategy or any(not 0<=i<a.shape[0] for i in req.row_strategy) or any(not 0<=j<a.shape[1] for j in req.col_strategy):
        raise HTTPException(422,'Invalid periodic strategy sequence')
    from safe_numeric import json_number
    import numpy as np
    totals=np.zeros(2,dtype=np.longdouble); trace=[]; weight=np.longdouble(1)
    for t in range(req.rounds):
        i=req.row_strategy[t%len(req.row_strategy)];j=req.col_strategy[t%len(req.col_strategy)]
        payoff=[float(a[i,j]),float(b[i,j])]
        totals=totals+weight*np.asarray(payoff,dtype=np.longdouble)
        if req.rounds<=200:trace.append({'round':t+1,'actions':[i,j],'payoffs':payoff,'discounted_cumulative':[json_number(z) for z in totals]})
        weight*=req.discount
    return {'discounted_payoffs':[json_number(z) for z in totals],'rounds':req.rounds,'trace':trace,'trace_truncated':req.rounds>200,'note':'Open-loop periodic action sequences; no contingent strategies or equilibrium claim'}
