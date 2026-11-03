"""Reproducible multi-start fictitious play for finite two-player games.

This is a heuristic, not a complete Nash-equilibrium algorithm. Every returned
profile includes a directly computed unilateral-deviation (Nash gap) certificate.
"""
import numpy as np
from fastapi import HTTPException


def nash_gap(a, b, x, y):
    from safe_numeric import json_number
    xx=np.asarray(x,dtype=np.longdouble)
    yy=np.asarray(y,dtype=np.longdouble)
    aa=np.asarray(a,dtype=np.longdouble)
    bb=np.asarray(b,dtype=np.longdouble)
    row=xx@aa@yy;col=xx@bb@yy
    row_regret=max(np.longdouble(0),np.max(aa@yy)-row)
    col_regret=max(np.longdouble(0),np.max(xx@bb)-col)
    return {'row_regret':json_number(row_regret),'col_regret':json_number(col_regret),
            'epsilon':json_number(max(row_regret,col_regret)),
            'expected_payoffs':[json_number(row),json_number(col)]}


def approximate_nash(a, b, iterations=10000, restarts=8, seed=0):
    if not 100 <= iterations <= 200000 or not 1 <= restarts <= 64:
        raise HTTPException(422, 'iterations must be 100..200000 and restarts 1..64')
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if iterations*restarts > 500_000:
        raise HTTPException(422,'Iteration-restart work budget exceeded (500,000)')
    rng = np.random.default_rng(seed)
    m, n = a.shape
    # Positive affine payoff transformations preserve best responses.
    # Long-double intermediates keep the finite float64 range, including ±1e308.
    al=np.asarray(a,dtype=np.longdouble);bl=np.asarray(b,dtype=np.longdouble)
    scale_a=np.ptp(al) or np.longdouble(1)
    scale_b=np.ptp(bl) or np.longdouble(1)
    aa=np.asarray((al - np.min(al))/scale_a,dtype=float)
    bb=np.asarray((bl - np.min(bl))/scale_b,dtype=float)
    best = None
    for restart in range(restarts):
        cx = np.ones(m) if restart == 0 else rng.random(m) + 0.01
        cy = np.ones(n) if restart == 0 else rng.random(n) + 0.01
        # Cumulative scores allow O(m+n) best-response updates per iteration.
        row_scores = aa @ cy
        col_scores = cx @ bb
        for step in range(iterations):
            i = int(np.argmax(row_scores + rng.uniform(0, 1e-12, m)))
            j = int(np.argmax(col_scores + rng.uniform(0, 1e-12, n)))
            cx[i] += 1.0
            cy[j] += 1.0
            row_scores += aa[:, j]
            col_scores += bb[i, :]
            if (step + 1) % 50 == 0 or step == iterations - 1:
                x = cx / cx.sum()
                y = cy / cy.sum()
                certificate = nash_gap(a, b, x, y)
                if best is None or np.longdouble(certificate['epsilon']) < np.longdouble(best['certificate']['epsilon']):
                    best = {'row_probabilities': x.tolist(), 'col_probabilities': y.tolist(),
                            'certificate': certificate, 'restart': restart, 'iteration': step + 1}
    return {'profile': best, 'method': 'multi-start simultaneous fictitious play',
            'iterations': iterations, 'restarts': restarts, 'seed': seed,
            'guaranteed_equilibrium': False,
            'note': 'Heuristic profile; epsilon is verified in original payoff units, not a convergence guarantee.'}
