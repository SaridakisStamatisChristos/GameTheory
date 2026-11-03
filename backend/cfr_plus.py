"""CFR+ with exact (bounded) pure best-response exploitability certificate.

Assumes a finite two-player zero-sum perfect-recall game. Terminal payoffs are
for player zero; player one receives their negation.
"""
from best_response import best_response
from math import isfinite
from fastapi import HTTPException
from pydantic import BaseModel, Field

class CFRPlusRequest(BaseModel):
    nodes: dict[str, dict]
    root: str = 'root'
    iterations: int = Field(default=1000, ge=1, le=20000)
    max_br_profiles: int = Field(default=100000, ge=1, le=1000000)

def validate_cfr_game(req: CFRPlusRequest):
    nodes=req.nodes
    if len(nodes)>2000 or req.root not in nodes: raise HTTPException(422,'Invalid root or oversized game')
    if len(nodes)*req.iterations>2_000_000:
        raise HTTPException(422,'CFR+ node-iteration budget exceeded (2,000,000)')
    infos={}; seen=set(); visiting=set(); histories={0:{},1:{}}
    def inspect(k, h0, h1):
        if k not in nodes or k in visiting or k in seen: raise HTTPException(422,'Graph must be a rooted tree without shared nodes or cycles')
        visiting.add(k); n=nodes[k]
        if 'payoff' in n:
            if not isinstance(n['payoff'],(int,float)) or isinstance(n['payoff'],bool) or not isfinite(n['payoff']): raise HTTPException(422,'Invalid payoff')
        else:
            acts=n.get('actions'); p=n.get('player')
            if not isinstance(acts,dict) or not acts or any(not isinstance(a,str) or not a or not isinstance(v,str) for a,v in acts.items()): raise HTTPException(422,'Invalid actions')
            if p=='chance':
                pr=n.get('probabilities')
                if not isinstance(pr,dict) or set(pr)!=set(acts) or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not isfinite(x) or x<0 for x in pr.values()) or abs(sum(pr.values())-1)>1e-9: raise HTTPException(422,'Invalid chance distribution')
            elif type(p) is int and p in (0,1):
                i=n.get('infoset')
                if not isinstance(i,str) or not i: raise HTTPException(422,'Missing infoset')
                signature=(p,tuple(sorted(acts)))
                if i in infos and infos[i]!=signature: raise HTTPException(422,'Inconsistent infoset (including player ownership)')
                infos[i]=signature
                history=h0 if p==0 else h1
                if i in histories[p] and histories[p][i]!=history: raise HTTPException(422,'Perfect recall violation')
                histories[p][i]=history
            else: raise HTTPException(422,'Invalid player')
            for a,child in acts.items():
                nh0=h0+((n['infoset'],a),) if p==0 else h0
                nh1=h1+((n['infoset'],a),) if p==1 else h1
                inspect(child,nh0,nh1)
        visiting.remove(k);seen.add(k)
    try:
        inspect(req.root,(),())
    except RecursionError as exc:
        raise HTTPException(422, 'Game tree exceeds supported recursion depth') from exc
    if len(seen)!=len(nodes):raise HTTPException(422,'Unreachable nodes')
    return infos

def solve_cfr_plus(req: CFRPlusRequest):
    infos = validate_cfr_game(req)
    nodes = req.nodes
    regrets={i:{a:0. for a in acts} for i,(_,acts) in infos.items()}
    sums={i:{a:0. for a in acts} for i,(_,acts) in infos.items()}
    def policy(i):
        r=regrets[i]; total=sum(r.values())
        return {a:(r[a]/total if total>0 else 1/len(r)) for a in r}
    def walk(k,r0,r1,ch,up,pol,delta,weights,t):
        n=nodes[k]
        if 'payoff' in n:return float(n['payoff'])
        p=n['player'];acts=n['actions']
        if p=='chance':return sum(n['probabilities'][a]*walk(c,r0,r1,ch*n['probabilities'][a],up,pol,delta,weights,t) for a,c in acts.items())
        i=n['infoset'];s=pol[i]
        vals={a:walk(c,r0*(s[a] if p==0 else 1),r1*(s[a] if p==1 else 1),ch,up,pol,delta,weights,t) for a,c in acts.items()}
        v=sum(s[a]*vals[a] for a in acts)
        if p==up:
            cf=ch*(r1 if p==0 else r0); own=r0 if p==0 else r1
            for a in acts:
                delta[i][a]+=cf*(vals[a]-v)*(1 if p==0 else -1)
                weights[i][a]+=t*ch*own*s[a]
        return v
    for t in range(1,req.iterations+1):
        for up in (0,1):
            pol={i:policy(i) for i in infos}
            delta={i:{a:0. for a in acts} for i,(_,acts) in infos.items()}
            weights={i:{a:0. for a in acts} for i,(_,acts) in infos.items()}
            walk(req.root,1.,1.,1.,up,pol,delta,weights,t)
            for i in infos:
                for a in regrets[i]:
                    regrets[i][a]=max(0.,regrets[i][a]+delta[i][a])
                    sums[i][a]+=weights[i][a]
    avg={i:{a:(v/sum(s.values()) if sum(s.values()) else 1/len(s)) for a,v in s.items()} for i,s in sums.items()}
    def evaluate(k,pol):
        n=nodes[k]
        if 'payoff' in n:return float(n['payoff'])
        weights=n['probabilities'] if n['player']=='chance' else pol[n['infoset']]
        return sum(weights[a]*evaluate(c,pol) for a,c in n['actions'].items())
    value=evaluate(req.root,avg)
    result={'method':'alternating CFR+ with linear averaging','iterations':req.iterations,'average_strategies':avg,'row_expected_payoff':value,'exploitability':None,'certificate_status':'not_computed'}
    br=[]
    for p in (0,1):
        payoff, _ = best_response(nodes, req.root, avg, p)
        br.append(payoff)
    result['best_response_row_payoff']=br[0]
    result['best_response_col_payoff_to_row']=br[1]
    result['exploitability']=max(0.,br[0]-br[1])
    result['certificate_status']='exact_perfect_recall_tree_best_response'
    return result
