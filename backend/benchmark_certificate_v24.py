"""Deterministic independent certificate audit of the pivot solver."""
import json, time
from pathlib import Path
import numpy as np
from independent_lh import solve_independent_lh
from numerical_certificate import certify_bimatrix


def run(seed=2401, repeats=3):
    rng=np.random.default_rng(seed)
    records=[]
    for n in (2,3,5,8):
        for family in ('integer','zero_sum','scaled'):
            for k in range(5):
                a=rng.integers(-8,9,(n,n)).astype(float)
                b=rng.integers(-8,9,(n,n)).astype(float)
                if family=='zero_sum': b=-a
                if family=='scaled': a*=1e-4; b*=1e4
                durations=[]; certificates=[]; failures=[]
                for _ in range(repeats):
                    start=time.perf_counter_ns()
                    result=solve_independent_lh(a,b,max_labels=min(3,2*n))
                    durations.append((time.perf_counter_ns()-start)/1e6)
                    failures.append(len(result['failures']))
                    certificates.append([certify_bimatrix(a,b,e['row_probabilities'],e['col_probabilities']) for e in result['equilibria']])
                records.append({'size':n,'family':family,'case':k,'median_ms':float(np.median(durations)),
                                'valid_equilibria':len([c for c in certificates[0] if c['valid']]),
                                'invalid_certificates':sum(not c['valid'] for group in certificates for c in group),
                                'failed_labels':failures})
    return {'seed':seed,'repeats':repeats,'records':records}


if __name__=='__main__':
    result=run()
    Path('benchmark_certificate_v24.json').write_text(json.dumps(result,indent=2))
    rows=result['records']
    print('cases',len(rows),'valid_cases',sum(r['valid_equilibria']>0 for r in rows),
          'invalid_certificates',sum(r['invalid_certificates'] for r in rows),
          'median_ms',round(float(np.median([r['median_ms'] for r in rows])),4))
