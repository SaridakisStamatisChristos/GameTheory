"""v59 seeded numerical support-root campaign with independent rational Nash checks."""
import argparse
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
import platform
import statistics
import time

import numpy as np
from multiplayer_mixed import solve_multiplayer_mixed


def fixtures(seed=59059):
    rng=np.random.default_rng(seed)
    cases=[]
    m=np.array([[1,-1],[-1,1]],dtype=int)
    matching=np.zeros((3,2,2,1),dtype=int)
    matching[0,:,:,0]=m
    matching[1,:,:,0]=-m
    cases.append(('matching_pennies_dummy',matching))
    parity=np.zeros((3,2,2,2),dtype=int)
    for player in range(3):
        for profile in product(range(2),repeat=3):
            parity[(player,*profile)]=(1 if profile[player]==
                (profile[(player+1)%3]^profile[(player+2)%3]) else -1)
    cases.append(('three_player_parity',parity))
    coord=np.zeros((3,2,2,2),dtype=int)
    for action in range(2):
        coord[(slice(None),action,action,action)]=2
    cases.append(('three_player_coordination',coord))
    dominant=np.zeros((3,2,2,2),dtype=int)
    for player in range(3):
        for profile in product(range(2),repeat=3):
            dominant[(player,*profile)]=int(profile[player]==0)
    cases.append(('three_player_dominance',dominant))
    cases.append(('four_player_all_ties',np.zeros((4,2,2,2,2),dtype=int)))
    for shape in ((2,2,2),(2,2,3),(2,3,2),(2,2,2,2),(2,2,2,2,2)):
        games=rng.integers(-4,5,size=(len(shape),*shape))
        cases.append((f'random_{"x".join(map(str,shape))}',games))
    return cases


def independently_audit(payoffs,candidate):
    """Exact rational original-game payoff audit without sharing the solver's code."""
    profile_shape=payoffs.shape[1:]
    n=len(profile_shape)
    probabilities=[[F(q) for q in row] for row in candidate['strategies_exact']]
    if (len(probabilities)!=n or any(len(row)!=actions or sum(row)!=1 or
           any(x<0 for x in row) for row,actions in zip(probabilities,profile_shape))):
        raise AssertionError('Non-simplex multiplayer candidate')
    regrets=[]
    for player in range(n):
        values=[]
        for pure_action in range(profile_shape[player]):
            payoff=F(0)
            for profile in product(*(range(actions) for actions in profile_shape)):
                if profile[player]!=pure_action:
                    continue
                opp=F(1)
                for other in range(n):
                    if player!=other:
                        opp *= probabilities[other][profile[other]]
                payoff+=opp*F(str(float(payoffs[(player,*profile)])))
            values.append(payoff)
        expected=sum(probabilities[player][a]*values[a]
                     for a in range(profile_shape[player]))
        regrets.append(max(values)-expected)
    epsilon=max(regrets)
    if epsilon != F(candidate['epsilon_exact']) or epsilon<0:
        raise AssertionError('Independent rational incentive audit mismatch')
    if [F(v) for v in candidate['player_unilateral_regrets_exact']] != regrets:
        raise AssertionError('Per-player deviation gain incorrect')
    return epsilon


def independently_pure(payoffs):
    shape=payoffs.shape[1:]
    return {pr for pr in product(*(range(k) for k in shape))
            if all(payoffs[(p,*pr)]>=payoffs[(p,*(pr[:p]+(a,)+pr[p+1:]))]
                   for p in range(len(shape)) for a in range(shape[p]))}


def run(repetitions=3, seed=59059):
    if type(repetitions) is not int or not 1<=repetitions<=10:
        raise ValueError('repetitions must be an integer in 1..10')
    games=fixtures(seed)
    digest=sha256(json.dumps([(name,u.tolist()) for name,u in games],
                   separators=(',',':')).encode()).hexdigest()
    data=[]
    for name,payoffs in games:
        durations=[]
        signatures=[]
        last=None
        for _ in range(repetitions):
            start=time.perf_counter()
            out=solve_multiplayer_mixed(payoffs,seed=seed,
                     starts_per_pattern=3,max_evaluations=5000,
                     max_support_patterns=96,max_candidates=20)
            durations.append(time.perf_counter()-start)
            signatures.append([(c['support'],c['strategies_exact'],c['epsilon_exact'])
                              for c in out['mixed_candidates']])
            if last is not None and signatures[-1] != signatures[0]:
                raise AssertionError(f'Nondeterministic support candidates: {name}')
            last=out
        if {tuple(c['actions']) for c in last['pure_nash_equilibria']} != independently_pure(payoffs):
            raise AssertionError(f'Incorrect exhaustive pure Nash set: {name}')
        max_epsilon=F(0)
        for candidate in last['mixed_candidates']:
            epsilon=independently_audit(payoffs,candidate)
            if epsilon>F(last['requested_epsilon_tolerance_exact']):
                raise AssertionError(f'Invalid rational Nash epsilon: {name}')
            max_epsilon=max(max_epsilon,epsilon)
        if last['mixed_nash_complete']:
            raise AssertionError('Numerical root search is never exhaustive')
        if last['function_evaluations']>5000:
            raise AssertionError('Search exceeded its evaluation budget')
        if name=='matching_pennies_dummy' and not any(
                c['status']=='certified_exact_rational_nash' and
                c['support']==[[0,1],[0,1],[0]] for c in last['mixed_candidates']):
            raise AssertionError('Known three-player mixed Nash not discovered')
        if name=='three_player_parity' and not any(
                c['support']==[[0,1],[0,1],[0,1]] for c in last['mixed_candidates']):
            raise AssertionError('Known fully mixed 3-player Nash not discovered')
        data.append({'case':name,'players':payoffs.shape[0],
                     'action_counts':list(payoffs.shape[1:]),
                     'pure_nash_count':len(last['pure_nash_equilibria']),
                     'mixed_candidates_found':last['mixed_candidates_found'],
                     'all_support_patterns_attempted':last['all_support_patterns_attempted'],
                     'support_patterns_examined':last['support_patterns_examined'],
                     'support_patterns_total':last['support_patterns_total'],
                     'root_calls':last['root_calls'],
                     'function_evaluations':last['function_evaluations'],
                     'incomplete_reason':last['incomplete_reason'],
                     'maximum_exact_rational_epsilon':str(max_epsilon),
                     'independent_rational_deviation_verified':True,
                     'independent_pure_nash_sets_equal':True,
                     'median_seconds':statistics.median(durations),
                     'all_seconds':durations})
    return {'release':'v59','seed':seed,'repetitions':repetitions,
            'fixtures_sha256':digest,'python':platform.python_version(),
            'numpy':np.__version__,
            'cases':data,
            'limitations':'Numerical support roots may miss equilibria or entire supports; rational regret verification certifies only returned profiles. No all-mixed-Nash-completeness, global runtime SOTA, or independent N-player Gambit comparator claimed.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--seed',type=int,default=59059)
    parser.add_argument('--output',default='benchmark_multiplayer_mixed_v59_results.json')
    args=parser.parse_args()
    out=run(args.repetitions,args.seed)
    with open(args.output,'w',encoding='utf-8') as f:
        json.dump(out,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({'cases_checked':len(out['cases']),
                      'known_mixed_games_found':all(
                          c['mixed_candidates_found']>0 for c in out['cases'][:2]),
                      'independent_certificates_all_valid':all(
                          c['independent_rational_deviation_verified'] for c in out['cases']),
                      'output':args.output},indent=2))


if __name__=='__main__':
    main()
