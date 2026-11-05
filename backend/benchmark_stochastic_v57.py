"""Reproducible v57 discounted zero-sum stochastic game correctness campaign."""
import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
import numpy as np
from stochastic_zero_sum import solve_discounted_zero_sum_stochastic


def fixtures(seed=57057):
    rng=np.random.default_rng(seed)
    games=[]
    def add(name,r,p,discount,tolerance=1e-6,max_iterations=250,expected=None):
        games.append((name,np.asarray(r,dtype=float),np.asarray(p,dtype=float),
                      discount,tolerance,max_iterations,expected))
    add('one_state_matching',[[[1,-1],[-1,1]]],np.ones((1,2,2,1)),.9,expected=[0])
    add('one_state_coordination',[[[2,0],[0,1]]],np.ones((1,2,2,1)),
        .7,expected=[(2/3)/.3])
    p=np.asarray([[.8,.2],[.3,.7]])
    rr=np.asarray([2.,-3.])
    add('two_state_policy_free_mdp',rr[:,None,None],p[:,None,None,:],
        .8,expected=np.linalg.solve(np.eye(2)-.8*p,rr).tolist())
    two=np.asarray([[[1,-1],[-1,1]],[[2,0],[0,1]]],dtype=float)
    transition=np.asarray([[.6,.4],[.2,.8]])
    add('two_state_fixed_transition_mixed',two,
        np.broadcast_to(transition[:,None,None,:],(2,2,2,2)),
        .6,expected=np.linalg.solve(np.eye(2)-.6*transition,[0,2/3]).tolist())
    ring=np.asarray([[0,1,0],[0,0,1],[1,0,0]])
    rr3=np.asarray([1.,-1.,2.])
    add('three_state_deterministic_cycle',rr3[:,None,None],ring[:,None,None,:],
        .75,expected=np.linalg.solve(np.eye(3)-.75*ring,rr3).tolist())
    for states,actions,discount in ((2,2,.6),(3,2,.65),(3,3,.75),(4,3,.7)):
        reward=rng.integers(-5,6,size=(states,actions,actions)).astype(float)
        transitions=rng.dirichlet(np.ones(states),size=(states,actions,actions))
        add(f'random_{states}_state_{actions}x{actions}',reward,transitions,discount)
    add('partial_budget',np.asarray([[[1.]]]),np.asarray([[[[1.]]]]),
        .98,tolerance=1e-10,max_iterations=3,expected=[50.])
    return games


def run(repetitions=3,seed=57057):
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError('Repetitions must be an integer from 1 to 10')
    games=fixtures(seed)
    canonical=[(name,r.tolist(),p.tolist(),gamma,tol,it,expected)
               for name,r,p,gamma,tol,it,expected in games]
    digest=hashlib.sha256(json.dumps(canonical,sort_keys=True,
                                    separators=(',',':')).encode()).hexdigest()
    records=[]
    for name,r,p,gamma,tol,it,expected in games:
        durations=[];last=None
        for _ in range(repetitions):
            start=time.perf_counter()
            result=solve_discounted_zero_sum_stochastic(
                r,p,discount=gamma,tolerance=tol,max_iterations=it)
            durations.append(time.perf_counter()-start)
            if last is not None and (
                    last['status'] != result['status'] or
                    last['value_estimates'] != result['value_estimates'] or
                    last['row_stationary_policy'] != result['row_stationary_policy']):
                raise AssertionError('Stochastic iterations are nondeterministic')
            last=result
        if name == 'partial_budget':
            if last['tolerance_met'] or last['status'] != 'partial_bound':
                raise AssertionError('Truncated iteration concealed a partial result')
        elif not last['tolerance_met']:
            raise AssertionError(f'Expected bounded convergence for {name}')
        reference_error=None
        if expected is not None:
            reference_error=float(np.max(np.abs(np.asarray(last['value_estimates'])-expected)))
            if reference_error > last['value_error_bound']+1e-8:
                raise AssertionError(f'Independent analytic/Markov reference outside bound: {name}')
        records.append({
            'case':name,'states':r.shape[0],
            'row_actions':r.shape[1],'col_actions':r.shape[2],
            'discount':gamma,'tolerance':tol,'iteration_cap':it,
            'status':last['status'],'tolerance_met':last['tolerance_met'],
            'iterations':last['iterations'],'stage_games_solved':last['stage_games_solved'],
            'max_state_saddle_gap':max(last['state_saddle_gaps']),
            'max_bellman_residual_bound':last['max_bellman_residual_bound'],
            'value_error_bound':last['value_error_bound'],
            'stationary_policy_regret_bound':last['stationary_policy_regret_bound'],
            'reference_abs_error':reference_error,
            'median_seconds':statistics.median(durations),
            'all_seconds':durations})
    return {
        'research_gate':'v57 bounded Shapley iteration for discounted zero-sum stochastic games',
        'seed':seed,'repetitions':repetitions,'fixtures_sha256':digest,
        'python':sys.version.split()[0],'platform':platform.platform(),
        'numpy':np.__version__,
        'independent_references':'one-state known zero-sum matrix values and direct linear Markov reward systems',
        'cases':records,
        'limitations':'Bounds rely on floating-point stage security intervals, not outward-rounded formal interval arithmetic. No independent stochastic-game library or broad cross-engine speed baseline yet.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--seed',type=int,default=57057)
    parser.add_argument('--output',default='benchmark_stochastic_v57_results.json')
    args=parser.parse_args(argv)
    out=run(args.repetitions,args.seed)
    with open(args.output,'w',encoding='utf8') as handle:
        json.dump(out,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps({
        'cases_checked':len(out['cases']),
        'reference_errors_within_bounds':all(
            c['reference_abs_error'] is None or
            c['reference_abs_error'] <= c['value_error_bound']+1e-8
            for c in out['cases']),
        'partial_budget_exposed':out['cases'][-1]['status']=='partial_bound',
        'output':args.output},indent=2))


if __name__=='__main__':
    main()
