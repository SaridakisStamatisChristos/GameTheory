"""Reproducible paired latency and Nash certificate benchmark."""
import json, time, statistics
import numpy as np
from independent_lh import solve_independent_lh
from lemke_howson import solve_lemke_howson

def run():
    rng=np.random.default_rng(20261103)
    records=[]
    for size in (2,3,4,5):
        for case in range(10):
            a=rng.integers(-8,9,(size,size)).astype(float)
            b=rng.integers(-8,9,(size,size)).astype(float)
            for name,fn in [('independent',solve_independent_lh),('nashpy',solve_lemke_howson)]:
                times=[]; result=None
                for repeat in range(3):
                    t=time.perf_counter_ns(); result=fn(a,b); times.append((time.perf_counter_ns()-t)/1e6)
                records.append({'size':size,'case':case,'solver':name,'median_ms':statistics.median(times),'equilibria':len(result['equilibria']),'failures':len(result['failures']),'max_gap':max((e['nash_gap'] for e in result['equilibria']),default=None)})
    with open('benchmark_independent_lh_results.json','w') as f: json.dump(records,f,indent=2)
    for size in (2,3,4,5):
        for name in ('independent','nashpy'):
            subset=[r for r in records if r['size']==size and r['solver']==name]
            print(size,name,'median_ms',round(statistics.median(r['median_ms'] for r in subset),3),'valid',sum(bool(r['equilibria']) for r in subset),'/10')
if __name__=='__main__':run()
