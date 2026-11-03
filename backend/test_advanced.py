import numpy as np
from advanced import support_equilibria,replicator

def test_matching_pennies():
    a=np.array([[1.,-1.],[-1.,1.]])
    eq=support_equilibria(a,-a)
    assert len(eq)==1
    assert np.allclose(eq[0]['row_probabilities'],[.5,.5])

def test_battle_of_sexes():
    a=np.array([[2.,0.],[0.,1.]])
    b=np.array([[1.,0.],[0.,2.]])
    assert len(support_equilibria(a,b))==3

def test_evolution_simplex():
    result=replicator([[1,0],[0,1]],steps=40)
    assert all(abs(sum(x)-1)<1e-10 and min(x)>=0 for x in result['trajectory'])
