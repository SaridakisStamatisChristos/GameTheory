"""Deterministic adversarial equilibrium campaign; correctness separated from timing."""
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
from hybrid_nash import solve_hybrid
from lemke_howson import solve_lemke_howson


def cases(seed=20261103, random_per_size=12):
    rng = np.random.default_rng(seed)
    for n in (2, 3, 4, 5):
        z = np.zeros((n,n))
        yield f'constant_{n}', z, z
        a = np.ones((n,n)); b = np.ones((n,n))
        yield f'identical_{n}', a, b
        a = np.eye(n); yield f'coordination_{n}', a, a
        a = rng.normal(size=(n,n)); yield f'zero_sum_{n}', a, -a
        for j in range(random_per_size):
            a = rng.integers(-4, 5, size=(n,n)).astype(float)
            b = rng.integers(-4, 5, size=(n,n)).astype(float)
            yield f'integer_{n}_{j}', a, b


def certify(a,b,eq,tol=1e-6):
    x=np.asarray(eq['row_probabilities'],dtype=float)
    y=np.asarray(eq['col_probabilities'],dtype=float)
    if x.shape!=(a.shape[0],) or y.shape!=(a.shape[1],): return None
    if not (np.isfinite(x).all() and np.isfinite(y).all()): return None
    if min(x.min(),y.min()) < -tol or abs(x.sum()-1)>tol or abs(y.sum()-1)>tol: return None
    return max(0.,float(np.max(a@y)-x@a@y),float(np.max(x@b)-x@b@y))


def run(seed=20261103, random_per_size=12, repeats=2):
    if repeats<1 or random_per_size<0: raise ValueError('Invalid repetitions')
    records=[]
    for name,a,b in cases(seed,random_per_size):
        for solver_name,solver in [('hybrid',lambda:solve_hybrid(a,b,max_pairs=1000)),('lemke_howson',lambda:solve_lemke_howson(a,b))]:
            times=[]; failures=[]; gaps=[]
            for _ in range(repeats):
                start=time.perf_counter_ns()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',RuntimeWarning)
                        result=solver()
                    times.append((time.perf_counter_ns()-start)/1e6)
                    certified=[v for eq in result['equilibria'] if (v:=certify(a,b,eq)) is not None and v<=1e-6]
                    if certified: gaps.append(min(certified))
                    else: failures.append('no certified equilibrium')
                except Exception as exc:
                    times.append((time.perf_counter_ns()-start)/1e6)
                    failures.append(type(exc).__name__+': '+str(exc)[:160])
            records.append({'case':name,'shape':list(a.shape),'solver':solver_name,'median_ms':float(np.median(times)),
                            'valid_runs':len(gaps),'failures':failures,'best_gap':min(gaps) if gaps else None})
    return {'seed':seed,'random_per_size':random_per_size,'repeats':repeats,'records':records,
            'limitations':'Local non-isolated Python timings; no independent SOTA claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--random-per-size',type=int,default=12);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--output',default='benchmark_adversarial_results.json');args=p.parse_args()
    Path(args.output).write_text(json.dumps(run(args.seed,args.random_per_size,args.repeats),indent=2))
