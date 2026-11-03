"""Reproducible paired latency distributions and independently checked equilibria.

Run: python benchmark_distributions.py --repeats 15 --output benchmark_distributions.json
Benchmark order alternates to reduce systematic warm-cache/order bias.
"""
import argparse
import json
import platform
import statistics
import time
import warnings
from pathlib import Path
import numpy as np
from hybrid_nash import solve_hybrid
from lemke_howson import solve_lemke_howson
from reference_benchmark import cases, gap


def summary(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'median_ms': float(np.median(a)),
            'p10_ms': float(np.percentile(a, 10)), 'p90_ms': float(np.percentile(a, 90)),
            'min_ms': float(a.min()), 'max_ms': float(a.max()),
            'mean_ms': float(a.mean()), 'std_ms': float(a.std(ddof=1)) if len(a)>1 else 0.0}


def benchmark(repeats=15, warmups=2, max_pairs=100000):
    if repeats < 1 or warmups < 0:
        raise ValueError('repeats must be positive; warmups nonnegative')
    records=[]
    for name, a, b in cases():
        solvers = {
            'hybrid': lambda: solve_hybrid(a,b,max_pairs=max_pairs),
            'lemke_howson': lambda: solve_lemke_howson(a,b),
        }
        times={k:[] for k in solvers}
        valid={k:0 for k in solvers}
        failures={k:0 for k in solvers}
        best_gap={k:None for k in solvers}
        for i in range(warmups+repeats):
            keys=list(solvers)
            if i % 2: keys.reverse()
            for key in keys:
                start=time.perf_counter_ns()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        result=solvers[key]()
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    gaps=[gap(a,b,e['row_probabilities'],e['col_probabilities']) for e in result['equilibria']]
                    if i>=warmups:
                        times[key].append(elapsed)
                        if any(g<=1e-6 for g in gaps): valid[key]+=1
                        else: failures[key]+=1
                        if gaps:
                            g=min(gaps)
                            best_gap[key]=g if best_gap[key] is None else min(g,best_gap[key])
                except Exception:
                    if i>=warmups:
                        times[key].append((time.perf_counter_ns()-start)/1e6)
                        failures[key]+=1
        records.append({'case':name,'shape':list(a.shape), 'solvers':{
            k:{'latency':summary(times[k]), 'valid_runs':valid[k],
               'failed_or_no_equilibrium_runs':failures[k], 'best_nash_gap':best_gap[k]}
            for k in solvers}})
    return {'environment':{'python':platform.python_version(),'numpy':np.__version__,
            'platform':platform.platform()}, 'repeats':repeats,'warmups':warmups,
            'timing':'wall-clock perf_counter_ns, paired alternating order',
            'validity':'independent unilateral-deviation Nash gap <= 1e-6',
            'limitations':'Python wall-clock timings; no CPU affinity or isolated load; solver failures counted; not SOTA evidence.',
            'cases':records}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repeats',type=int,default=15)
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--output',default='benchmark_distributions.json')
    args=p.parse_args()
    Path(args.output).write_text(json.dumps(benchmark(args.repeats,args.warmups),indent=2))
