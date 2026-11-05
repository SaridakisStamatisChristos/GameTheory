"""v60 adversarial completeness, exact quadratic fields and independent payoff audits."""
from fractions import Fraction as F
from itertools import product
from math import sqrt

import numpy as np
import pytest
from fastapi.testclient import TestClient

from multiplayer_polynomial_v60 import (
    Quadratic, _quadratic_roots, _sqrt_rational, _sign, _encode,
    _difference_coefficients, _difference, _verify_point, _point,
    _one_mixed, _two_mixed, _three_mixed, solve_certified_binary_multiplayer,
)
from multiplayer_exact import _game
from main import app


def make_game(coefficients):
    """Player-action1 payoff minus action0 payoff equals a+b*y+c*z+d*y*z."""
    u=np.zeros((3,2,2,2),dtype=float)
    for p,(a,b,c,d) in enumerate(coefficients):
        others=[q for q in range(3) if q!=p]
        for profile in product((0,1),repeat=3):
            if profile[p]:
                y,z=(profile[q] for q in others)
                u[(p,*profile)]=float(a+b*y+c*z+d*y*z)
    return u


def canonical_two_roots():
    # F0=y-z, F1=x+z-1, F2=x*y-1/8:
    # x=1-z, y=z, z(1-z)=1/8; two algebraic fully mixed points.
    return make_game([(0,1,-1,0),(-1,1,1,0),(-F(1,8),0,0,1)])


def root_game(constant):
    # F0=y-z, F1=x-z, F2=x*y+constant.
    return make_game([(0,1,-1,0),(0,1,-1,0),(constant,0,0,1)])


def _pair(x,discriminant):
    if x['type']=='rational':
        return (F(x['value']),F(0))
    assert x['type']=='quadratic' and F(x['radicand'])==discriminant
    return F(x['a']),F(x['b'])


def _add(x,y):
    return x[0]+y[0],x[1]+y[1]


def _mul(x,y,d):
    return x[0]*y[0]+x[1]*y[1]*d,x[0]*y[1]+x[1]*y[0]


def _independent_full_mixed_oracle(game,equilibrium):
    """Independent direct product-profile algebraic field arithmetic."""
    encoded=equilibrium['action1_probabilities_exact']
    radicals={F(v['radicand']) for v in encoded if v['type']=='quadratic'}
    assert len(radicals)<=1
    d=next(iter(radicals),F(0))
    x=[_pair(v,d) for v in encoded]
    for p in range(3):
        assert equilibrium['support'][p]==[0,1]
        approx=float(x[p][0])+float(x[p][1])*sqrt(float(d))
        assert 0<approx<1
        action=[]
        for a in range(2):
            value=F(0),F(0)
            for profile in product((0,1),repeat=3):
                if profile[p]!=a:
                    continue
                weight=(F(1),F(0))
                for other in range(3):
                    if other!=p:
                        q=x[other] if profile[other] else (F(1)-x[other][0],-x[other][1])
                        weight=_mul(weight,q,d)
                value=_add(value,_mul(weight,(F(str(float(game[(p,*profile)]))),F(0)),d))
            action.append(value)
        assert action[0]==action[1]


def test_exact_algebraic_two_fully_mixed_roots_and_completeness():
    game=canonical_two_roots()
    out=solve_certified_binary_multiplayer(game)
    assert out['complete_nash_set'],out['unresolved_support_cells']
    assert out['status']=='complete_exact_semialgebraic_nash_set'
    assert out['support_cells_processed']==out['support_cells_total']==27
    assert out['unvisited_support_cells']==0
    full=[p for p in out['isolated_equilibria']
          if p['support']==[[0,1]]*3]
    assert len(full)==2
    assert all(any(z['type']=='quadratic' for z in e['action1_probabilities_exact'])
               for e in full)
    for result in full:
        _independent_full_mixed_oracle(game,result)


def test_rational_single_full_mixed_solution():
    out=solve_certified_binary_multiplayer(root_game(-F(1,4)))
    assert out['complete_nash_set']
    full=[x for x in out['isolated_equilibria'] if x['support']==[[0,1]]*3]
    assert len(full)==1
    assert full[0]['action1_probabilities_exact']==[
        {'type':'rational','value':'1/2'}]*3


def test_double_root_exact_and_negative_discriminant():
    repeated=make_game([(0,1,-1,0),(0,1,-1,0),
                        (F(1,4),-1,0,1)])
    solution=solve_certified_binary_multiplayer(repeated)
    assert solution['complete_nash_set']
    assert len([x for x in solution['isolated_equilibria']
                if x['support']==[[0,1]]*3])==1
    no_full=solve_certified_binary_multiplayer(root_game(1))
    assert no_full['complete_nash_set']
    assert not any(x['support']==[[0,1]]*3 for x in no_full['isolated_equilibria'])


def test_linear_and_constant_eliminants():
    linear=make_game([(0,1,-1,0),(0,1,-1,0),(-F(1,2),1,0,0)])
    out=solve_certified_binary_multiplayer(linear)
    assert out['complete_nash_set']
    assert any(x['support']==[[0,1]]*3 for x in out['isolated_equilibria'])
    constant=make_game([(0,1,-1,0),(0,1,-1,0),(1,0,0,0)])
    out=solve_certified_binary_multiplayer(constant)
    assert out['complete_nash_set']
    assert not any(x['support']==[[0,1]]*3 for x in out['isolated_equilibria'])


def test_singular_fully_mixed_eliminant_flags_incompleteness():
    parity=np.zeros((3,2,2,2),dtype=float)
    for p in range(3):
        for profile in product((0,1),repeat=3):
            parity[(p,*profile)] = (1 if profile[p] ==
                (profile[(p+1)%3]^profile[(p+2)%3]) else -1)
    out=solve_certified_binary_multiplayer(parity)
    assert not out['complete_nash_set']
    assert out['status']=='partial_exact_support_decomposition'
    assert out['unresolved_support_cells'][0]['reason'] == 'identically_zero_eliminant'
    assert out['unresolved_support_cells'][0]['support_pattern']==['m']*3
    assert len(out['unresolved_support_cells'][0]['exact_indifference_coefficients'])==3


def test_true_singular_denominator_root_does_not_claim_complete():
    # z=1/2 makes F1=(z-1/2)(x-1/2) indifferent for all x;
    # the eliminant zero at that singular z is not safely divisible.
    u=make_game([(0,1,-1,0),
                 (F(1,4),-F(1,2),-F(1,2),1),
                 (-F(1,4),1,0,0)])
    out=solve_certified_binary_multiplayer(u)
    assert not out['complete_nash_set']
    assert out['unresolved_support_cells'][0]['reason']=='singular_denominator_root'


def test_identically_zero_denominator_is_conservatively_unresolved():
    u=make_game([(-F(1,2),0,1,0),(0,1,-1,0),
                 (-F(1,8),0,0,1)])
    out=solve_certified_binary_multiplayer(u)
    assert not out['complete_nash_set']
    assert out['unresolved_support_cells'][0]['reason']=='nonisolated_elimination_denominator'


def test_partial_budget_reports_unvisited_cells_no_false_completeness():
    u=canonical_two_roots()
    result=solve_certified_binary_multiplayer(u,max_cells=8)
    assert result['support_cells_processed']==8
    assert result['unvisited_support_cells']==19
    assert not result['complete_nash_set']
    assert not result['isolated_equilibria'] or all(len(eq['support'])==3
                                                   for eq in result['isolated_equilibria'])
    no_full=solve_certified_binary_multiplayer(u,max_cells=26)
    assert not no_full['complete_nash_set']
    assert no_full['unvisited_support_cells']==1
    assert not any(eq['support']==[[0,1]]*3 for eq in no_full['isolated_equilibria'])


def test_one_mixed_open_interval_and_isolated_rational_endpoint():
    z=np.zeros((3,2,2,2))
    out=solve_certified_binary_multiplayer(z)
    assert not out['complete_nash_set']
    intervals=[f for f in out['continuous_equilibrium_cells']
               if f['kind']=='one_parameter_interval']
    assert intervals
    assert any(f['action1_probability_lower_exact']=='0'
               and f['action1_probability_upper_exact']=='1'
               and not f['lower_inclusive'] and not f['upper_inclusive']
               for f in intervals)
    # Player0 indifferent, player1 pure0 requires x<=1/2,
    # player2 pure1 requires x>=1/2.
    u=make_game([(0,0,0,0),(-F(1,2),1,0,0),(-F(1,2),1,0,0)])
    out=solve_certified_binary_multiplayer(u)
    pts=[e for e in out['isolated_equilibria']
         if e['support']==[[0,1],[0],[1]]]
    assert len(pts)==1
    assert pts[0]['action1_probabilities_exact'][0]['value']=='1/2'


def test_one_mixed_rational_closed_inner_interval_and_impossible_interval():
    feasible=make_game([(0,0,0,0),(-F(7,10),1,0,0),
                        (-F(3,10),1,0,0)])
    out=solve_certified_binary_multiplayer(feasible)
    interval=next(f for f in out['continuous_equilibrium_cells']
                  if f['support']==[[0,1],[0],[1]])
    assert interval['action1_probability_lower_exact']=='3/10'
    assert interval['action1_probability_upper_exact']=='7/10'
    assert interval['lower_inclusive'] and interval['upper_inclusive']
    impossible=make_game([(0,0,0,0),(-F(1,5),1,0,0),
                          (-F(4,5),1,0,0)])
    out=solve_certified_binary_multiplayer(impossible)
    assert not any(f['support']==[[0,1],[0],[1]]
                   for f in out['continuous_equilibrium_cells'])
    assert not any(p['support']==[[0,1],[0],[1]]
                   for p in out['isolated_equilibria'])


def test_two_mixed_unique_rational_point_and_continuous_bilinear_cell():
    u=make_game([(-F(1,3),1,0,0),(-F(2,3),1,0,0),(1,0,0,0)])
    out=solve_certified_binary_multiplayer(u)
    point=next(v for v in out['isolated_equilibria']
               if v['support']==[[0,1],[0,1],[1]])
    assert [x['value'] for x in point['action1_probabilities_exact']]==[
        '2/3','1/3','1']
    # Two free probabilities, inequality x*y-1/2 >= 0.
    free=make_game([(0,0,0,0),(0,0,0,0),(-F(1,2),0,0,1)])
    out=solve_certified_binary_multiplayer(free)
    region=next(f for f in out['continuous_equilibrium_cells']
                if f['support']==[[0,1],[0,1],[1]])
    assert region['kind']=='bilinear_semialgebraic_cell'
    assert region['free_action1_probability_players']==[0,1]
    assert region['pure_best_response_polynomial_ge_zero_exact']=={
        'constant':'-1/2','p0':'0','p1':'0','p0_p1':'1'}


def test_two_mixed_weak_boundary_corner_is_not_falsely_feasible():
    u=make_game([(0,0,0,0),(0,0,0,0),(0,0,0,-1)])
    out=solve_certified_binary_multiplayer(u)
    assert not any(e['support']==[[0,1],[0,1],[1]]
                   for e in out['isolated_equilibria'])
    assert not any(f['support']==[[0,1],[0,1],[1]]
                   for f in out['continuous_equilibrium_cells'])


def test_two_mixed_one_free_and_two_fixed_root_outcomes():
    onefree=make_game([(0,0,0,0),(-F(1,2),1,0,0),
                       (-F(1,2),0,1,0)])
    out=solve_certified_binary_multiplayer(onefree)
    region=next(f for f in out['continuous_equilibrium_cells']
                if f['support']==[[0,1],[0,1],[1]])
    assert region['fixed_action1_probabilities_exact']=={'0':'1/2'}
    assert region['free_action1_probability_players']==[1]
    too_large=make_game([(-2,1,0,0),(-F(1,2),1,0,0),(1,0,0,0)])
    out=solve_certified_binary_multiplayer(too_large)
    assert not any(p['support']==[[0,1],[0,1],[1]]
                   for p in out['isolated_equilibria'])


def test_two_mixed_constant_nonzero_indifference_rejects():
    u=make_game([(1,0,0,0),(0,0,0,0),(0,0,0,0)])
    out=solve_certified_binary_multiplayer(u)
    assert not any(f['support']==[[0,1],[0,1],[1]]
                   for f in out['continuous_equilibrium_cells'])


def test_quadratic_field_exact_sign_and_arithmetic_branches():
    q=Quadratic(1,1,F(2))
    assert _sign(q)==1 and _sign(-q)==-1
    assert _sign(Quadratic(0,1,2))==1
    assert _sign(Quadratic(0,-1,2))==-1
    assert _sign(Quadratic(1,0,2))==1
    assert _sign(Quadratic(0,0,2))==0
    assert _sign(Quadratic(-2,1,2))==-1
    assert _sign(Quadratic(2,-1,2))==1
    assert _sign(Quadratic(-1,1,2))==1
    assert _sign(Quadratic(1,-1,2))==-1
    assert _sign((q+F(3))-(q*2))==_sign(3-q)
    assert _sign(1/q)>0
    assert _sign(q/q-1)==0
    assert _sign((q*2)-(q+q))==0
    assert _encode(Quadratic(2,0,2))=={'type':'rational','value':'2'}
    assert _encode(F(1,3))=={'type':'rational','value':'1/3'}
    with pytest.raises(ValueError,match='Negative'):
        Quadratic(0,1,-2)
    with pytest.raises(ValueError,match='Incompatible'):
        _=q+Quadratic(0,1,3)
    with pytest.raises(ZeroDivisionError):
        _=q/Quadratic(0,0,2)


def test_polynomial_root_cases_and_exact_square_detection():
    assert _sqrt_rational(F(9,16))==F(3,4)
    assert _sqrt_rational(F(2)) is None
    assert _quadratic_roots([F(1),F(0),F(0)])==[]
    assert _quadratic_roots([F(-1),F(2),F(0)])==[F(1,2)]
    assert _quadratic_roots([F(1),F(0),F(1)])==[]
    assert _quadratic_roots([F(1,4),F(-1),F(1)])==[F(1,2)]
    assert _quadratic_roots([F(-1,4),F(0),F(1)])==[F(-1,2),F(1,2)]
    assert len(_quadratic_roots([F(-1,8),F(0),F(1)]))==2


def test_direct_exact_point_guard_rejects_non_nash_and_bad_simplex():
    game=canonical_two_roots()
    _,profiles,u=_game(game)
    assert _point([F(-1),F(0),F(0)],u,profiles) is None
    assert _point([F(2),F(0),F(0)],u,profiles) is None
    assert _point([F(1,2)]*3,u,profiles) is None
    diff=_difference_coefficients(u,profiles)
    assert len(diff)==3
    assert isinstance(_difference(diff[0],0,[F(1,2)]*3),F)


@pytest.mark.parametrize('budget',[0,-1,28,True,1.1])
def test_invalid_cell_budget(budget):
    with pytest.raises(ValueError,match='max_cells'):
        solve_certified_binary_multiplayer(canonical_two_roots(),max_cells=budget)


def test_three_player_binary_scope_rejected_for_other_shapes():
    with pytest.raises(ValueError,match='exactly three'):
        solve_certified_binary_multiplayer(np.zeros((3,2,2,1)))
    with pytest.raises(ValueError,match='exactly three'):
        solve_certified_binary_multiplayer(np.zeros((4,2,2,2,2)))
    with pytest.raises(ValueError,match='Expected'):
        solve_certified_binary_multiplayer(np.zeros((2,2,2)))


def test_http_polynomial_completeness_and_partial():
    game=canonical_two_roots().tolist()
    with TestClient(app,raise_server_exceptions=False) as client:
        resp=client.post('/api/multiplayer/mixed/polynomial-exact',json={
            'payoffs':game})
        assert resp.status_code==200,resp.text
        result=resp.json()
        assert result['complete_nash_set']
        assert result['support_cells_processed']==27
        assert len([e for e in result['isolated_equilibria']
                    if e['support']==[[0,1]]*3])==2
        partial=client.post('/api/multiplayer/mixed/polynomial-exact',
                            json={'payoffs':game,'max_cells':1})
        assert partial.status_code==200
        assert not partial.json()['complete_nash_set']
        bad=client.post('/api/multiplayer/mixed/polynomial-exact',
                        json={'payoffs':game,'max_cells':28})
        assert bad.status_code==422
        wrong_shape=client.post('/api/multiplayer/mixed/polynomial-exact',
                                json={'payoffs':np.zeros((3,2,2,1)).tolist()})
        assert wrong_shape.status_code==422
        earlier=client.post('/api/multiplayer/exact',json={
            'payoffs':game,'mode':'pure'})
        assert earlier.status_code==200


def test_one_sided_fully_mixed_corner_sign_screen_is_exact():
    # Three-player strictly full support gives each opponent corner
    # positive weight. One weakly signed nonzero difference cannot tie.
    weak=make_game([(0,0,0,1),(0,1,-1,0),(0,1,-1,0)])
    out=solve_certified_binary_multiplayer(weak)
    assert out['complete_nash_set']
    assert not any(x['support']==[[0,1]]*3 for x in out['isolated_equilibria'])


def test_zero_norm_quadratic_field_and_early_denominator_case():
    # Defensive algebraic field representation also handles D=1, though
    # actual solver only creates it for nonsquare discriminants.
    assert _sign(Quadratic(1,-1,1))==0
    # z=1/2 kills denominator of x-equation but numerator is nonzero.
    # The eliminant can have that root; it must be skipped, not unresolved.
    impossible=make_game([(0,1,-1,0),(1,0,-F(1,2),1),
                          (-F(1,2),0,1,0)])
    out=solve_certified_binary_multiplayer(impossible)
    assert out['complete_nash_set']
    assert not out['unresolved_support_cells']


def test_out_of_support_full_mixed_root_is_discarded():
    # x=2 for any z, y=z; eliminant may have an interior z root,
    # but no strict-support Nash point has x=2.
    u=make_game([(0,1,-1,0),(-2,1,0,0),(-F(1,8),0,0,1)])
    out=solve_certified_binary_multiplayer(u)
    assert out['complete_nash_set']
    assert not any(x['support']==[[0,1]]*3 for x in out['isolated_equilibria'])


def test_internal_point_verifiers_defend_against_solver_corruption(monkeypatch):
    import multiplayer_polynomial_v60 as module
    from itertools import product as iter_product
    original=module._point
    try:
        _,profiles,u=_game(make_game([(0,0,0,0),(-F(1,2),1,0,0),
                                      (-F(1,2),1,0,0)]))
        coeffs=module._difference_coefficients(u,profiles)
        monkeypatch.setattr(module,'_point',lambda *_args:None)
        with pytest.raises(AssertionError,match='one-mixed interval'):
            module._one_mixed(('m','0','1'),coeffs,u,profiles)
        _,profiles,u=_game(make_game([(-F(1,3),1,0,0),
                                      (-F(2,3),1,0,0),(1,0,0,0)]))
        coeffs=module._difference_coefficients(u,profiles)
        with pytest.raises(AssertionError,match='two-mixed point'):
            module._two_mixed(('m','m','1'),coeffs,u,profiles)
        _,profiles,u=_game(canonical_two_roots())
        coeffs=module._difference_coefficients(u,profiles)
        with pytest.raises(AssertionError,match='non-Nash point'):
            module._three_mixed(coeffs,u,profiles)
    finally:
        monkeypatch.setattr(module,'_point',original)
