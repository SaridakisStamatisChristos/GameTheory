"""v58 reproducible multi-player exact Nash/CE correctness and timing campaign.

External numeric HiGHS reference solves the same recommendation-obedience
LP, independently rebuilt using tensor indices rather than the exact engine.
"""
import argparse
import hashlib
from itertools import product
import json
import platform
import statistics
import sys
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from multiplayer_exact import solve_exact_multiplayer


def fixtures(seed=58058):
    rng = np.random.default_rng(seed)
    games = []
    coordination = np.zeros((3,2,2,2), dtype=int)
    for action in range(2):
        coordination[(slice(None),action,action,action)] = 2
    games.append(('three_player_coordination',coordination))
    matrix = np.array([[1,-1],[-1,1]],dtype=int)
    no_pure = np.zeros((3,2,2,1),dtype=int)
    no_pure[0,:,:,0] = matrix
    no_pure[1,:,:,0] = -matrix
    games.append(('three_player_no_pure_matching',no_pure))
    games.append(('four_player_all_ties',np.zeros((4,2,2,2,2),dtype=int)))
    games.append(('five_player_singletons',np.array([1,2,3,4,5],
                  dtype=int).reshape((5,1,1,1,1,1))))
    for shape in ((2,2,2),(2,2,3),(2,2,2,2),(1,2,2,1,2)):
        for run in range(2):
            games.append((f'random_{"x".join(map(str,shape))}_{run}',
                          rng.integers(-5,6,size=(len(shape),*shape))))
    return games


def independent_reference(payoffs):
    shape = payoffs.shape[1:]
    profiles = list(product(*(range(s) for s in shape)))
    rows = []
    for p, actions in enumerate(shape):
        for advised in range(actions):
            for alternative in range(actions):
                if alternative == advised:
                    continue
                row = []
                for profile in profiles:
                    if profile[p] == advised:
                        replaced = (*profile[:p], alternative, *profile[p+1:])
                        row.append(int(payoffs[(p,*replaced)] -
                                       payoffs[(p,*profile)]))
                    else:
                        row.append(0)
                rows.append(row)
    objective = [-sum(int(payoffs[(p,*profile)])
                      for p in range(len(shape)))
                 for profile in profiles]
    result = linprog(objective,
                     A_ub=np.asarray(rows) if rows else None,
                     b_ub=np.zeros(len(rows)) if rows else None,
                     A_eq=np.ones((1,len(profiles))),b_eq=[1],
                     bounds=(0,None),method='highs')
    if not result.success:
        raise AssertionError('Independent HiGHS correlated LP failed')
    return -float(result.fun)


def independent_pure(payoffs):
    shape=payoffs.shape[1:]
    return {profile for profile in product(*(range(k) for k in shape))
            if all(payoffs[(p,*profile)] >=
                   payoffs[(p,*(profile[:p]+(alt,)+profile[p+1:]))]
                   for p in range(len(shape))
                   for alt in range(shape[p]))}


def compare_case(name,payoffs,repetitions):
    samples=[]
    values=[]
    for _ in range(repetitions):
        start=perf_counter()
        out=solve_exact_multiplayer(payoffs, max_operations=2000000)
        samples.append(perf_counter()-start)
        if not out['correlated_equilibrium_computed']:
            raise AssertionError('Required exact correlated result missing')
        if not out['pure_nash_complete']:
            raise AssertionError('Exhaustive pure Nash claim missing')
        if not out['maximum_obedience_gain_exact']=='0':
            raise AssertionError('Correlated recommendation violated')
        pure={tuple(point['actions']) for point in out['pure_nash_equilibria']}
        if pure!=independent_pure(payoffs):
            raise AssertionError('Incorrect multiplayer pure Nash set')
        values.append(out['objective_value_exact'])
    if len(set(values))!=1:
        raise AssertionError('Non-deterministic rational objective')
    reference=independent_reference(payoffs)
    ours=float(values[0]) if '/' not in values[0] else float(
        __import__('fractions').Fraction(values[0]))
    error=abs(ours-reference)
    if error>1e-8:
        raise AssertionError('Exact multiplayer welfare differs from independent HiGHS')
    return {'case':name,'players':payoffs.shape[0],
            'shape':list(payoffs.shape[1:]),
            'pure_nash_count':len(pure),
            'obedience_inequalities_checked':out['obedience_inequalities_checked'],
            'welfare_value_exact':values[0],
            'independent_highs_welfare':reference,
            'independent_highs_abs_error':error,
            'exact_operations':out['exact_operations'],
            'median_seconds':statistics.median(samples),
            'all_seconds':samples,'independent_pure_sets_equal':True}


def run(repetitions=3,seed=58058):
    if type(repetitions) is not int or not 1<=repetitions<=10:
        raise ValueError('Repetitions must be integer 1..10')
    games=fixtures(seed)
    canonical=json.dumps([(name,u.tolist()) for name,u in games],
                         separators=(',',':')).encode()
    result=[compare_case(name,u,repetitions) for name,u in games]
    return {'research_gate':'v58 3–5 player exact pure Nash and optimal CE',
            'seed':seed,'repetitions':repetitions,
            'fixtures_sha256':hashlib.sha256(canonical).hexdigest(),
            'python':sys.version.split()[0],
            'platform':platform.platform(),
            'numpy':np.__version__,
            'independent_reference':'SciPy HiGHS numerical LP on original welfare objective and independently constructed recommendation-obedience constraints; direct pure best-response enumeration',
            'cases':result,
            'limitations':'Small bounded games, floating HiGHS tolerance, not large-scale runtime or general mixed Nash completeness evidence.'}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--seed',type=int,default=58058)
    parser.add_argument('--output',default='benchmark_multiplayer_v58_results.json')
    args=parser.parse_args(argv)
    outcome=run(args.repetitions,args.seed)
    with open(args.output,'w',encoding='utf8') as f:
        json.dump(outcome,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({'cases_checked':len(outcome['cases']),
                      'max_highs_welfare_error':max(
                          c['independent_highs_abs_error'] for c in outcome['cases']),
                      'pure_sets_equal':all(
                          c['independent_pure_sets_equal'] for c in outcome['cases']),
                      'output':args.output},indent=2))


if __name__=='__main__':
    main()
