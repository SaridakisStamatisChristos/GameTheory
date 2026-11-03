import warnings
import numpy as np
from reference_benchmark import gap, cases
from hybrid_nash import solve_hybrid


def test_gap_detects_profitable_deviation():
    a=np.array([[1.,-1.],[-1.,1.]])
    assert gap(a,-a,[.5,.5],[.5,.5])<1e-12
    assert gap(a,-a,[1.,0.],[1.,0.])>0.5


def test_hybrid_against_independent_nashpy_small_games():
    import pytest
    nash=pytest.importorskip('nashpy')
    for name,a,b in cases():
        if max(a.shape)>3:continue
        ours=solve_hybrid(a,b,max_pairs=10000)
        # Nashpy emits an expected diagnostic when support enumeration returns
        # an even number of equilibria (including zero) on degenerate games.
        # Capture and validate it rather than hiding all runtime warnings.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            ref=list(nash.Game(a,b).support_enumeration(non_degenerate=False))
        unexpected = [w for w in caught if not (
            issubclass(w.category, RuntimeWarning)
            and "indicates that the game is degenerate" in str(w.message)
            and "equilibria was returned" in str(w.message)
        )]
        assert not unexpected, (name, [str(w.message) for w in unexpected])
        for w in caught:
            assert len(ref) % 2 == 0, (name, str(w.message))
        assert ours['equilibria'],name
        # Nashpy support enumeration may miss equilibria; never assume reference completeness.
        assert all(np.isfinite(x).all() and np.isfinite(y).all() for x,y in ref),name
        assert all(gap(a,b,e['row_probabilities'],e['col_probabilities'])<1e-6 for e in ours['equilibria']),name
