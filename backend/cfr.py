"""Full-tree chance-sampled? No: exact vanilla CFR for finite two-player zero-sum games.
Information sets must have perfect recall; caller is responsible for this modeling assumption.
"""
from collections import defaultdict
from math import isfinite
from fastapi import HTTPException
from pydantic import BaseModel, Field

class CFRRequest(BaseModel):
    nodes: dict[str, dict]
    root: str = 'root'
    iterations: int = Field(default=1000, ge=1, le=20000)


def solve_cfr(req: CFRRequest):
    nodes=req.nodes
    if len(nodes)>2000 or req.root not in nodes:
        raise HTTPException(422,'Invalid root or tree too large')
    visiting=set(); visited=set(); info_actions={}; info_player={}
    def inspect(k):
        if k not in nodes or k in visiting: raise HTTPException(422,'Missing node or cycle')
        if k in visited: raise HTTPException(422,'Graph must be a tree without shared nodes')
        visiting.add(k); node=nodes[k]
        if 'payoff' in node:
            p=node['payoff']
            if not isinstance(p,(int,float)) or not isfinite(p): raise HTTPException(422,'Invalid terminal payoff')
        else:
            actions=node.get('actions')
            if not isinstance(actions,dict) or not actions: raise HTTPException(422,'Nonterminal requires actions')
            player=node.get('player')
            if player=='chance':
                probs=node.get('probabilities',{})
                if set(probs)!=set(actions) or any(not isinstance(v,(int,float)) or not isfinite(v) or v<0 for v in probs.values()) or abs(sum(probs.values())-1)>1e-8:
                    raise HTTPException(422,'Invalid chance probabilities')
            elif player in (0,1):
                info=node.get('infoset')
                if not isinstance(info,str) or not info: raise HTTPException(422,'Missing information set')
                if info in info_actions and (info_actions[info]!=tuple(actions) or info_player[info]!=player):
                    raise HTTPException(422,'Inconsistent information set')
                info_actions[info]=tuple(actions);info_player[info]=player
            else: raise HTTPException(422,'Invalid player')
            for child in actions.values(): inspect(child)
        visiting.remove(k);visited.add(k)
    try:
        inspect(req.root)
    except RecursionError as exc:
        raise HTTPException(422,'Game tree exceeds supported recursion depth') from exc
    if len(visited)!=len(nodes): raise HTTPException(422,'Unreachable nodes')
    if len(nodes)*req.iterations > 2_000_000:
        raise HTTPException(422,'CFR node-iteration budget exceeded (2,000,000)')
    regrets={k:defaultdict(float) for k in info_actions}
    strategy_sums={k:defaultdict(float) for k in info_actions}
    def strategy(info):
        acts=info_actions[info]; positive=[max(0.,regrets[info][a]) for a in acts]; total=sum(positive)
        return dict(zip(acts,[v/total for v in positive] if total>0 else [1/len(acts)]*len(acts)))
    def traverse(k, reach0,reach1,chance, update):
        node=nodes[k]
        if 'payoff' in node:return float(node['payoff'])
        acts=node['actions']; player=node['player']
        if player=='chance':return sum(float(node['probabilities'][a])*traverse(child,reach0,reach1,chance*float(node['probabilities'][a]),update) for a,child in acts.items())
        info=node['infoset']; s=strategy(info)
        values={a:traverse(child,reach0*(s[a] if player==0 else 1),reach1*(s[a] if player==1 else 1),chance,update) for a,child in acts.items()}
        value=sum(s[a]*values[a] for a in acts)
        if player==update:
            counterfactual=chance*(reach1 if player==0 else reach0)
            own_reach=reach0 if player==0 else reach1
            for a in acts:
                regrets[info][a]+=counterfactual*(values[a]-value)*(1 if player==0 else -1)
                strategy_sums[info][a]+=chance*own_reach*s[a]
        return value
    for _ in range(req.iterations):
        traverse(req.root,1.,1.,1.,0)
        traverse(req.root,1.,1.,1.,1)
    average={}
    for info,acts in info_actions.items():
        total=sum(strategy_sums[info].values())
        average[info]={a:(strategy_sums[info][a]/total if total>0 else 1/len(acts)) for a in acts}
    def expected(k):
        node=nodes[k]
        if 'payoff' in node:return float(node['payoff'])
        if node['player']=='chance': weights=node['probabilities']
        else:weights=average[node['infoset']]
        return sum(weights[a]*expected(child) for a,child in node['actions'].items())
    return {'average_strategies':average,'row_expected_payoff':expected(req.root),'iterations':req.iterations,'method':'full-tree alternating vanilla CFR','note':'Convergence guarantee requires finite two-player zero-sum extensive-form game with perfect recall; no exploitability certificate is asserted.'}
