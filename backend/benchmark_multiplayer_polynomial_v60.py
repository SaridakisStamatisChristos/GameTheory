"""v60 matched polynomial Nash enumeration cases with independent payoff checks."""
import argparse
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
from math import sqrt
import platform
import statistics
from time import perf_counter

import numpy as np
from multiplayer_polynomial_v60 import solve_certified_binary_multiplayer


def _game(coeff):
    u=np.zeros((3,2,2,2))
    for p,(a,b,c,d) in enumerate(coeff):
        others=[q for q in range(3) if q!=p]
        for profile in product((0,1),repeat=3):
            if profile[p]:
                y,z=(profile[q] for q in others)
                u[(p,*profile)]=float(a+b*y+c*z+d*y*z)
    return u


def fixtures(seed=60060):
    rng=np.random.default_rng(seed)
    cases=[
        ('two_irrational_fully_mixed',
         _game([(0,1,-1,0),(-1,1,1,0),(-F(1,8),0,0,1)]),2,True),
        ('one_rational_fully_mixed',
         _game([(0,1,-1,0),(0,1,-1,0),(-F(1,4),0,0,1)]),1,True),
        ('one_double_fully_mixed',
         _game([(0,1,-1,0),(0,1,-1,0),(F(1,4),-1,0,1)]),1,True),
        ('negative_discriminant',
         _game([(0,1,-1,0),(0,1,-1,0),(1,0,0,1)]),0,True),
        ('linear_eliminant',
         _game([(0,1,-1,0),(0,1,-1,0),(-F(1,2),1,0,0)]),1,True),
        ('four_vertex_coordination',
         np.asarray([[[[2 if i==j==k else 0 for k in range(2)]
                       for j in range(2)] for i in range(2)]]*3),None,None),
        ('dominant_actions',
         _game([(1,0,0,0),(-1,0,0,0),(1,0,0,0)]),0,True),
        ('all_ties_degenerate',np.zeros((3,2,2,2)),None,False),
    ]
    parity=np.zeros((3,2,2,2))
    for p in range(3):
        for profile in product((0,1),repeat=3):
            parity[(p,*profile)]=(1 if profile[p]==
                 (profile[(p+1)%3]^profile[(p+2)%3]) else -1)
    cases.append(('parity_degenerate',parity,None,False))
    cases.append(('singular_denominator',
                  _game([(0,1,-1,0),(F(1,4),-F(1,2),-F(1,2),1),
                         (-F(1,4),1,0,0)]),None,False))
    for i in range(8):
        cases.append((f'seeded_{i:02d}',
                      rng.integers(-5,6,size=(3,2,2,2)),None,None))
    return cases


def _number(z):
    if z['type']=='rational':
        return float(F(z['value']))
    return float(F(z['a'])) + float(F(z['b']))*sqrt(float(F(z['radicand'])))


def _independent_regret(game,point):
    probs=[_number(x) for x in point['action1_probabilities_exact']]
    if any(p< -1e-10 or p>1+1e-10 for p in probs):
        raise AssertionError('Polynomial solver returned non-simplex profile')
    max_regret=0.
    for player in range(3):
        action_values=[]
        for action in range(2):
            value=0.
            for profile in product((0,1),repeat=3):
                if profile[player]!=action:
                    continue
                weight=1.
                for p in range(3):
                    if p!=player:
                        weight*=probs[p] if profile[p] else 1-probs[p]
                value+=weight*float(game[(player,*profile)])
            action_values.append(value)
        expected=(1-probs[player])*action_values[0]+probs[player]*action_values[1]
        max_regret=max(max_regret,max(action_values)-expected)
    if max_regret>1e-8:
        raise AssertionError('Independent numerical original-payoff Nash check failed')
    return max_regret


def run(repetitions=3,seed=60060):
    if type(repetitions) is not int or not 1<=repetitions<=10:
        raise ValueError('repetitions must be an integer 1..10')
    cases=fixtures(seed)
    fixture_digest=sha256(json.dumps([(n,g.tolist()) for n,g,_,_ in cases],
                                separators=(',',':')).encode()).hexdigest()
    records=[]
    for name,game,expected_full,expected_complete in cases:
        elapsed=[];result=None
        for _ in range(repetitions):
            start=perf_counter()
            observed=solve_certified_binary_multiplayer(game)
            elapsed.append(perf_counter()-start)
            if result is not None and result!=observed:
                raise AssertionError(f'Non-deterministic exact support classification: {name}')
            result=observed
        if expected_complete is not None and result['complete_nash_set']!=expected_complete:
            raise AssertionError(f'Invalid completeness disclosure for {name}')
        full=[x for x in result['isolated_equilibria']
              if x['support']==[[0,1]]*3]
        if expected_full is not None and len(full)!=expected_full:
            raise AssertionError(f'Incorrect algebraic root count for {name}')
        error=max((_independent_regret(game,c) for c in result['isolated_equilibria']),default=0.)
        records.append({
            'case':name,'exact_completeness_claim':result['complete_nash_set'],
            'support_cells_total':result['support_cells_total'],
            'support_cells_processed':result['support_cells_processed'],
            'unresolved_support_cells':len(result['unresolved_support_cells']),
            'unresolved_reasons':[x['reason'] for x in result['unresolved_support_cells']],
            'isolated_equilibrium_points':len(result['isolated_equilibria']),
            'fully_mixed_algebraic_roots':len(full),
            'continuous_semialgebraic_cells':len(result['continuous_equilibrium_cells']),
            'maximum_independent_numerical_unilateral_regret':error,
            'median_seconds':statistics.median(elapsed),'all_seconds':elapsed})
    return {'release':'v60','seed':seed,'repetitions':repetitions,
            'fixtures_sha256':fixture_digest,
            'python':platform.python_version(),'numpy':np.__version__,
            'cases':records,
            'evidence':'exact rational/quadratic Nash certificates by direct original-payoff audit; independent numerical regret check and analytic quadratic-root counts',
            'limitations':'Only three-player two-action normal-form games; complete only if all 27 support cells classified. Singular fully-mixed polynomial systems remain unresolved. Timing is a host-dependent smoke experiment.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--seed',type=int,default=60060)
    parser.add_argument('--output',default='benchmark_multiplayer_polynomial_v60_results.json')
    args=parser.parse_args()
    out=run(args.repetitions,args.seed)
    with open(args.output,'w',encoding='utf8') as f:
        json.dump(out,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({
        'cases_checked':len(out['cases']),
        'all_27_support_cells_attempted':all(x['support_cells_processed']==27 for x in out['cases']),
        'two_irrational_roots_certified':out['cases'][0]['fully_mixed_algebraic_roots']==2,
        'degenerate_completeness_not_overclaimed':all(
            not out['cases'][i]['exact_completeness_claim'] for i in (7,8,9)),
        'output':args.output},indent=2))


if __name__=='__main__':
    main()
