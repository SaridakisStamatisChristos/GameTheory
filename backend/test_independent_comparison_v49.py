"""v49 independent Nashpy comparisons with equal normal-form input matrices."""
import numpy as np
import pytest
from benchmark_independent_nashpy_v49 import (
    fixtures, compare_case, run, _certify_numerical, _support_from_profile)


@pytest.mark.parametrize(('name','expected_families'),[
    ('matching_pennies',1),
    ('battle_of_the_sexes',3),
    ('prisoners_dilemma',1),
    ('rock_paper_scissors',1),
    ('coordination_three',7),
    ('all_ties_three',49),
])
def test_known_equilibria_and_nashpy_reference_membership(name,expected_families):
    cases={k:(a,b) for k,a,b in fixtures()}
    a,b=cases[name]
    result=compare_case(name,a,b,repetitions=1)
    assert result['exact_complete'] is True
    assert result['exact_support_families']==expected_families
    assert result['nashpy_supports_not_found_by_exact']==0
    assert result['nashpy_valid_profiles']>=1
    assert result['median_exact_seconds']>=0
    assert result['median_nashpy_seconds']>=0


def test_independent_seeded_random_games_reproduce_reference_membership():
    cases=fixtures(seed=49001)
    for name,a,b in cases:
        if name.startswith('random_'):
            result=compare_case(name,a,b,repetitions=1)
            assert result['exact_complete']
            assert result['nashpy_supports_not_found_by_exact']==0
            assert result['exact_support_families']>=1


def test_invalid_reference_profiles_are_not_silently_accepted():
    a=np.array([[1,-1],[-1,1]],dtype=float)
    b=-a
    assert _certify_numerical(a,b,[.5,.5],[.5,.5])
    assert not _certify_numerical(a,b,[1,0],[1,0])
    assert not _certify_numerical(a,b,[0,0],[1,0])
    assert not _certify_numerical(a,b,[float('nan'),0],[1,0])
    assert not _certify_numerical(a,b,[1],[.5,.5])
    assert _support_from_profile([1,0],[0,1])==((0,),(1,))


def test_comparator_input_budget_and_metadata_contract():
    name,a,b=fixtures()[0]
    with pytest.raises(ValueError):
        compare_case(name,a,b,repetitions=0)
    with pytest.raises(ValueError):
        compare_case(name,a,b,repetitions=21)
    report=run(repetitions=1,seed=49001)
    assert report['gambit']=='not_run_not_certified'
    assert report['open_spiel']=='not_run_not_certified'
    assert report['nashpy'] is not None
    assert len(report['cases'])==14
    assert all(c['exact_complete'] for c in report['cases'])
    assert all(c['nashpy_supports_not_found_by_exact']==0 for c in report['cases'])
