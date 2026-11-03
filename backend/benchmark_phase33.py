"""Reproducible exploratory independent LH timing and validity benchmark."""
import json
import platform
import statistics
import time
import numpy as np
from independent_lh import solve_independent_lh
from numerical_certificate import certify_bimatrix

rng = np.random.default_rng(3302)
results = []
for size in (2, 3, 5, 8, 10):
    samples = []
    valid = 0
    for case in range(8):
        a = rng.integers(-20, 21, (size, size)).astype(float)
        b = rng.integers(-20, 21, (size, size)).astype(float)
        solve_independent_lh(a, b, max_labels=min(4, 2 * size))  # warmup
        timings = []
        for _ in range(5):
            start = time.perf_counter_ns()
            outcome = solve_independent_lh(a, b, max_labels=min(4, 2 * size))
            timings.append((time.perf_counter_ns() - start) / 1e6)
            if not outcome['equilibria']:
                continue
            if all(certify_bimatrix(a,b,e['row_probabilities'],e['col_probabilities'])['valid'] for e in outcome['equilibria']):
                valid += 1
        samples.extend(timings)
    results.append({'size':size,'runs':len(samples),'valid_runs':valid,'median_ms':statistics.median(samples),'p90_ms':float(np.percentile(samples,90))})
print(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'seed':3302,'max_initial_labels':4,'results':results},indent=2))
