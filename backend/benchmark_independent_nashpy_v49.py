"""Reproducible independent Nashpy comparator for normal-form games.

This is a *verification* harness, not a claim of speed superiority. It
compares the complete rational-support geometry of GameTheory with
numerical point equilibria computed by independently maintained Nashpy.
A numerical point profile is not a description of a degenerate continuum.

From backend/: python benchmark_independent_nashpy_v49.py
"""
from fractions import Fraction
import json
import platform
import statistics
import sys
import time
import warnings
import numpy as np
import nashpy
import scipy
from general_exact_search import search_exact_nash_cells


def fixtures(seed=49001):
    rng=np.random.default_rng(seed)
    games=[
        ('matching_pennies',[[1,-1],[-1,1]],[[-1,1],[1,-1]]),
        ('battle_of_the_sexes',[[2,0],[0,1]],[[1,0],[0,2]]),
        ('prisoners_dilemma',[[3,0],[5,1]],[[3,5],[0,1]]),
        ('rock_paper_scissors',[[0,-1,1],[1,0,-1],[-1,1,0]],
                              [[0,1,-1],[-1,0,1],[1,-1,0]]),
        ('coordination_three',[[3,0,0],[0,2,0],[0,0,1]],
                              [[3,0,0],[0,2,0],[0,0,1]]),
        ('all_ties_three',np.zeros((3,3),dtype=int),
                          np.zeros((3,3),dtype=int)),
    ]
    for n in (2,3):
        for i in range(4):
            games.append((f'random_{n}_{i}',
                          rng.integers(-6,7,size=(n,n)),
                          rng.integers(-6,7,size=(n,n))))
    return [(name,np.asarray(a,dtype=float),np.asarray(b,dtype=float))
            for name,a,b in games]


def _support_from_profile(x,y,tol=1e-8):
    return (tuple(int(i) for i,v in enumerate(x) if v>tol),
            tuple(int(j) for j,v in enumerate(y) if v>tol))


def _certify_numerical(a,b,x,y):
    """Independently calculate simplex and unilateral-regret diagnostics."""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if (x.shape!=(a.shape[0],) or y.shape!=(a.shape[1],) or
        not np.isfinite(x).all() or not np.isfinite(y).all() or
        x.min() < -1e-8 or y.min() < -1e-8 or
        abs(x.sum()-1)>1e-8 or abs(y.sum()-1)>1e-8):
        return False
    p=float(x@a@y);q=float(x@b@y)
    eps=max(float(np.max(a@y)-p),float(np.max(x@b)-q),0.)
    return eps <= 1e-8


def compare_case(name,a,b,repetitions=3):
    if not 1<=repetitions<=20:
        raise ValueError('repetitions must be 1..20')
    exact_times=[];reference_times=[]
    exact=None;reference=None;warnings_seen=[]
    for _ in range(repetitions):
        start=time.perf_counter()
        out=search_exact_nash_cells(
            a,b,max_pairs=49,max_support=3,max_operations=500000,
            feasibility_backend='simplex',prune_supports=True,
            mixed_pruning=False)
        exact_times.append(time.perf_counter()-start)
        assert out['complete_equilibrium_set']
        if exact is not None and len(exact['families'])!=len(out['families']):
            raise AssertionError('Non-deterministic exact family count')
        exact=out
        start=time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            points=list(nashpy.Game(a,b).support_enumeration())
        reference_times.append(time.perf_counter()-start)
        warnings_seen.extend(str(w.message) for w in captured)
        reference=points
    exact_supports={
        (tuple(f['support']['row']),tuple(f['support']['col']))
        for f in exact['families']
    }
    accepted=[];invalid=0
    for x,y in reference:
        if _certify_numerical(a,b,x,y):
            accepted.append(_support_from_profile(x,y))
        else:
            invalid+=1
    reference_supports=set(accepted)
    # A validated reference NE must belong to an exact rational-support cell,
    # allowing for numerical support tolerance at tiny probabilities.
    missing=reference_supports-exact_supports
    if missing:
        raise AssertionError(f'Valid independent equilibria missed by exact search: {name}: {missing}')
    return {
        'case':name,'shape':list(a.shape),
        'exact_support_families':len(exact_supports),
        'nashpy_raw_profiles':len(reference),
        'nashpy_valid_profiles':len(accepted),
        'nashpy_invalid_or_unverified':invalid,
        'nashpy_supports_not_found_by_exact':len(missing),
        'nashpy_warnings':sorted(set(warnings_seen))[:6],
        'exact_complete':exact['complete_equilibrium_set'],
        'median_exact_seconds':statistics.median(exact_times),
        'median_nashpy_seconds':statistics.median(reference_times),
        'note':'Nashpy returns numerical profiles, not a full description of degenerate equilibrium continua',
    }


def run(repetitions=3,seed=49001):
    games=fixtures(seed)
    results=[compare_case(name,a,b,repetitions) for name,a,b in games]
    return {
        'comparison':'GameTheory exact strict-support geometry vs Nashpy numerical support enumeration',
        'seed':seed,'repetitions':repetitions,
        'python':sys.version.split()[0],'platform':platform.platform(),
        'numpy':np.__version__,'scipy':scipy.__version__,
        'nashpy':getattr(nashpy,'__version__','unreported'),
        'gambit':'not_run_not_certified',
        'open_spiel':'not_run_not_certified',
        'limitations':'No broad speedup claim; the algorithms output different representations and have different degeneracy behavior.',
        'cases':results,
    }


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
