"""Exact arithmetic boundaries and fail-closed algebra invariants."""
from fractions import Fraction
import pytest
import exact_support as ex

F=Fraction

def test_strict_vs_weak_exact_feasibility_and_work_limit():
    def run(rows,limit=100):
        return ex._strict_rational_feasible(rows,{'steps':0,'limit':limit})
    positive=[((F(1),),F(0),True),((F(-1),),F(-1),True)]
    assert run(positive)[0] == F(1,2)
    assert run([((F(1),),F(0),False),((F(-1),),F(0),False)])[0]==0
    assert run([((F(1),),F(0),True),((F(-1),),F(0),False)]) is None
    assert run([((F(0),),F(1),False)]) is None
    assert run([((),F(-1),False)])==[]
    assert run([((),F(0),True)]) is None
    with pytest.raises(ex.ExactWorkLimit):
        run(positive,limit=2)

def test_back_substitution_boundaries():
    choose=ex._choose_value
    assert choose([((F(1),),F(0),True)],[])==1
    assert choose([((F(1),),F(0),False)],[])==0
    assert choose([((F(-1),),F(0),True)],[])==-1
    assert choose([((F(-1),),F(0),False)],[])==0
    assert choose([],[])==0
    with pytest.raises(AssertionError):
        choose([((F(0),),F(1),False)],[])
    with pytest.raises(AssertionError):
        choose([((F(1),),F(1),True),((F(-1),),F(-1),False)],[])
    with pytest.raises(AssertionError):
        choose([((F(1),),F(2),False),((F(-1),),F(-1),False)],[])

def test_invalid_internal_witness_is_rejected(monkeypatch):
    assert not ex._verify_exact_equilibrium([[F(0)]],[[F(0)]],[F(-1)],[F(1)])
    monkeypatch.setattr(ex,'_strict_rational_feasible',lambda *args: [F(0)])
    with pytest.raises(AssertionError,match='probability witness'):
        ex._feasible_strategy([[F(0)]],(0,),(0,),{'steps':0,'limit':100})

def test_silent_incomplete_or_bad_certification_is_never_reported(monkeypatch):
    original=ex._feasible_strategy
    monkeypatch.setattr(ex,'_feasible_strategy',lambda *args:None)
    with pytest.raises(AssertionError,match='no exact Nash witness'):
        ex.solve_exact_support([[1]],[[1]])
    monkeypatch.setattr(ex,'_feasible_strategy',original)
    monkeypatch.setattr(ex,'_verify_exact_equilibrium',lambda *args:False)
    with pytest.raises(AssertionError,match='failed Nash certification'):
        ex.solve_exact_support([[1]],[[1]])

def test_invalid_budget_types():
    for kw in ({'max_pairs':0},{'max_pairs':226},{'max_pairs':True},
               {'max_operations':0},{'max_operations':500001},
               {'max_operations':True}):
        with pytest.raises(ValueError):
            ex.solve_exact_support([[1]],[[1]],**kw)
