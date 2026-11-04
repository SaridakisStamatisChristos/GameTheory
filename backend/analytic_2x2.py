"""Exact rational 2x2 best-response geometry for finite decimal payoff inputs.

The finite equilibria list preserves representative points for compatibility.
A separate exact-rational geometry field characterizes the entire equilibrium
set, including degenerate continuous families.
"""
from fractions import Fraction
import numpy as np
from safe_numeric import expected_payoffs
from exact_2x2_geometry import complete_nash_geometry


def solve_2x2(a,b,tol=1e-9):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=(2,2) or b.shape!=(2,2) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected two finite 2x2 payoff matrices')
    if not 0<tol<1:raise ValueError('Invalid tolerance')
    aa=[[Fraction(str(v)) for v in row] for row in a]
    bb=[[Fraction(str(v)) for v in row] for row in b]
    da0,da1=aa[0][0]-aa[1][0],aa[0][1]-aa[1][1]
    db0,db1=bb[0][0]-bb[0][1],bb[1][0]-bb[1][1]
    xs={Fraction(0),Fraction(1)}
    ys={Fraction(0),Fraction(1)}
    if db0!=db1:
        x=-db1/(db0-db1)
        if 0<=x<=1:xs.add(x)
    elif db0==0:xs.add(Fraction(1,2))
    if da0!=da1:
        y=-da1/(da0-da1)
        if 0<=y<=1:ys.add(y)
    elif da0==0:ys.add(Fraction(1,2))
    found=[]
    for x in sorted(xs):
        for y in sorted(ys):
            P=[x,1-x];Q=[y,1-y]
            row_options=[sum(aa[i][j]*Q[j] for j in range(2)) for i in range(2)]
            col_options=[sum(P[i]*bb[i][j] for i in range(2)) for j in range(2)]
            if max(row_options)!=sum(P[i]*row_options[i] for i in range(2)):
                continue
            if max(col_options)!=sum(Q[j]*col_options[j] for j in range(2)):
                continue
            xf=[float(z) for z in P]; yf=[float(z) for z in Q]
            found.append({'row_probabilities':xf,'col_probabilities':yf,
                          'expected_payoffs':expected_payoffs(a,b,xf,yf),
                          'nash_gap':0.,'certificate':'exact rational input 2x2 best responses'})
    components=complete_nash_geometry(aa,bb)
    return {'equilibria':found,'equilibrium_components':components,
            'geometry_complete':True,'method':'analytic_2x2',
            'complete':False,
            'note':'The finite equilibria list contains representative points, not the continuum. '
                   'The equilibrium_components union exactly characterizes the entire Nash set '
                   'of the decimal strings of the validated float64 input payoffs. '
                   'Probability interval endpoints are supplied as exact rational strings.'}
