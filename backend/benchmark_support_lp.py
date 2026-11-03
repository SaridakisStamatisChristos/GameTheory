"""Run: python benchmark_support_lp.py --output support-lp-results.json"""
import argparse,json,time
import numpy as np
from support_lp import solve_support_lp
from advanced import support_equilibria

def run(repeats=3,seed=1729):
    rng=np.random.default_rng(seed);out=[]
    cases=[('constant_2x3',np.zeros((2,3)),np.zeros((2,3))),('matching_pennies',np.array([[1.,-1],[-1,1]]),np.array([[-1.,1],[1,-1]]))]
    for n in (2,3,4):
        a=rng.normal(size=(n,n));b=rng.normal(size=(n,n));cases.append((f'random_general_{n}',a,b))
    for name,a,b in cases:
        for method,solver in [('legacy',support_equilibria),('support_lp',lambda x,y:solve_support_lp(x,y)['equilibria'])]:
            times=[];counts=[];gaps=[]
            for _ in range(repeats):
                start=time.perf_counter_ns();sols=solver(a,b);times.append((time.perf_counter_ns()-start)/1e6)
                counts.append(len(sols))
                for s in sols:
                    x=np.asarray(s['row_probabilities']);y=np.asarray(s['col_probabilities'])
                    gaps.append(max(0.,float(np.max(a@y)-x@a@y),float(np.max(x@b)-x@b@y)))
            out.append({'case':name,'algorithm':method,'median_ms':float(np.median(times)),
                        'p95_ms':float(np.percentile(times,95)),'equilibria':counts[0],
                        'max_gap':max(gaps) if gaps else None})
    return {'seed':seed,'repeats':repeats,'results':out,'caveat':'Small local samples; not a SOTA performance comparison.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='support-lp-results.json');p.add_argument('--repeats',type=int,default=3);args=p.parse_args()
    result=run(args.repeats)
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
