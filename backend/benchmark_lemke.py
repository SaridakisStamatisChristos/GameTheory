"""Reproducible multi-repeat cross-solver benchmark; run from backend directory."""
import json, time, statistics, platform
import numpy as np
from reference_benchmark import cases, gap
from hybrid_nash import solve_hybrid
from lemke_howson import solve_lemke_howson


def run(repeats=5):
    output=[]
    for name,a,b in cases():
        row={'case':name,'shape':list(a.shape),'solvers':{}}
        for label,fn in [('hybrid',lambda:solve_hybrid(a,b)),('lemke_howson',lambda:solve_lemke_howson(a,b))]:
            durations=[]; gaps=[]; failures=[]
            for _ in range(repeats):
                start=time.perf_counter_ns()
                try:
                    result=fn()
                    durations.append((time.perf_counter_ns()-start)/1e6)
                    gaps.extend(gap(a,b,e['row_probabilities'],e['col_probabilities']) for e in result['equilibria'])
                    failures.extend(result.get('failures',[]))
                except Exception as exc:
                    failures.append({'error':repr(exc)})
            row['solvers'][label]={'median_ms':statistics.median(durations) if durations else None,
                'min_ms':min(durations) if durations else None,'max_ms':max(durations) if durations else None,
                'best_nash_gap':min(gaps) if gaps else None,'failed_pivots_or_runs':len(failures),
                'valid_equilibrium_found':any(g<=1e-6 for g in gaps)}
        output.append(row)
    return {'repeats':repeats,'python':platform.python_version(),'numpy':np.__version__,
            'note':'Sequential warm-cache timings, no isolated CPU or statistical confidence intervals; not SOTA evidence.', 'cases':output}

if __name__=='__main__':
    from pathlib import Path
    result=run()
    Path('benchmark_lemke_results.json').write_text(json.dumps(result,indent=2))
    print('cases',len(result['cases']))
    for r in result['cases']:
        print(r['case'],*(f"{k}={v['median_ms']:.3f}ms" for k,v in r['solvers'].items() if v['median_ms'] is not None))
