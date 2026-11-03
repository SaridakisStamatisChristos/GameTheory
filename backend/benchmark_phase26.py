"""Deterministic, paired old/new pivot benchmark; run from backend directory."""
import json
import statistics
import time
from pathlib import Path
import numpy as np
from independent_lh import solve_independent_lh as new
from independent_lh_v21_baseline import solve_independent_lh as old
from numerical_certificate import certify_bimatrix

rng = np.random.default_rng(2601)
records = []
for n in (2, 3, 5, 8):
    for case in range(6):
        a = rng.integers(-10, 11, (n, n)).astype(float)
        b = rng.integers(-10, 11, (n, n)).astype(float)
        for name, fn in ((('new', new), ('old', old)) if case % 2 == 0 else (('old', old), ('new', new))):
            samples = []
            output = None
            for repeat in range(4):
                start = time.perf_counter_ns()
                output = fn(a, b, max_labels=min(3, 2*n))
                elapsed = (time.perf_counter_ns()-start)/1e6
                if repeat: samples.append(elapsed)
            valid = bool(output['equilibria']) and all(certify_bimatrix(a,b,e['row_probabilities'],e['col_probabilities'])['valid'] for e in output['equilibria'])
            records.append({'size':n,'case':case,'solver':name,'median_ms':statistics.median(samples),'valid':valid,'failures':len(output['failures'])})
summary = {str(n):{name:{'median_ms':round(statistics.median(r['median_ms'] for r in records if r['size']==n and r['solver']==name),4),'valid_cases':sum(r['valid'] for r in records if r['size']==n and r['solver']==name)} for name in ('old','new')} for n in (2,3,5,8)}
result={'summary':summary,'records':records,'repetitions_per_case':3,'seed':2601}
Path('benchmark_phase26_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
