"""Reproducible differential conditioning audit; run from backend directory."""
import argparse, json, time
from pathlib import Path
import numpy as np
from independent_lh import solve_independent_lh
from independent_lh_v21_baseline import solve_independent_lh as baseline


def cases(seed=20261103, per_family=12):
    rng=np.random.default_rng(seed)
    for n in (2, 3, 5, 8, 10):
        for family in ('integer','nearly_constant','asymmetric_scale','zero_sum','rank_one'):
            for k in range(per_family):
                a=rng.integers(-8,9,(n,n)).astype(float)
                b=rng.integers(-8,9,(n,n)).astype(float)
                if family=='nearly_constant':
                    a=2+1e-7*a; b=-3+1e-7*b
                elif family=='asymmetric_scale':
                    a*=1e-5; b*=1e5
                elif family=='zero_sum': b=-a
                elif family=='rank_one':
                    a=np.outer(rng.normal(size=n),rng.normal(size=n)); b=np.outer(rng.normal(size=n),rng.normal(size=n))
                yield n,family,k,a,b


def run(seed=20261103,per_family=12,max_labels=3):
    records=[]
    for n,family,k,a,b in cases(seed,per_family):
        item={'n':n,'family':family,'case':k}
        for name,solver in [('baseline',baseline),('optimized',solve_independent_lh)]:
            t=time.perf_counter_ns()
            try:
                result=solver(a,b,max_labels=min(max_labels,2*n))
                gaps=[]
                for eq in result['equilibria']:
                    x=np.asarray(eq['row_probabilities']);y=np.asarray(eq['col_probabilities'])
                    gaps.append(max(0.,float(np.max(a@y)-x@a@y),float(np.max(x@b)-x@b@y)))
                item[name]={'elapsed_ms':(time.perf_counter_ns()-t)/1e6,'valid_count':len(gaps),'max_gap':max(gaps) if gaps else None,'failed_labels':len(result['failures'])}
            except Exception as exc:
                item[name]={'elapsed_ms':(time.perf_counter_ns()-t)/1e6,'exception':repr(exc)}
        records.append(item)
    return {'seed':seed,'per_family':per_family,'max_labels':max_labels,'records':records}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--per-family',type=int,default=12);p.add_argument('--max-labels',type=int,default=3);p.add_argument('--output',default='benchmark_conditioning_v23.json');args=p.parse_args()
    data=run(per_family=args.per_family,max_labels=args.max_labels)
    Path(args.output).write_text(json.dumps(data,indent=2))
    for name in ('baseline','optimized'):
        vals=[r[name] for r in data['records']]
        durations=[v['elapsed_ms'] for v in vals]
        print(name,'cases',len(vals),'median_ms',round(float(np.median(durations)),4),'p90_ms',round(float(np.percentile(durations,90)),4),'with_equilibrium',sum(v.get('valid_count',0)>0 for v in vals),'exceptions',sum('exception' in v for v in vals))
