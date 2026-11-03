import json, statistics, time, platform
import numpy
from benchmark_kuhn_phase35 import kuhn_poker
from cfr_plus import CFRPlusRequest, solve_cfr_plus
from cfr_plus_compiled import solve_cfr_plus_compiled

def run():
    out=[]
    for iterations in (100, 500, 2000):
        req=CFRPlusRequest(nodes=kuhn_poker(),iterations=iterations)
        for name, fn in [('reference',solve_cfr_plus),('compiled',solve_cfr_plus_compiled)]:
            times=[]
            fn(req)
            for _ in range(5):
                start=time.perf_counter_ns(); result=fn(req)
                times.append((time.perf_counter_ns()-start)/1e6)
            out.append({'iterations':iterations,'solver':name,'median_ms':statistics.median(times),'times_ms':times,'exploitability':result['exploitability']})
    return {'python':platform.python_version(),'numpy':numpy.__version__,'results':out}
if __name__=='__main__':
    with open('benchmark_phase36_results.json','w') as f:json.dump(run(),f,indent=2)
