"""API error translation, storage boundaries and independent Nash checks."""
from types import SimpleNamespace
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
import main,research,support_lp
from best_response import best_response
from cfr_plus import CFRPlusRequest,validate_cfr_game

@pytest.fixture
def client():
    with TestClient(main.app,raise_server_exceptions=False) as c:yield c

@pytest.mark.parametrize('path,extras',[('/api/equilibria',{}),('/api/equilibria/hybrid',{'fast_support':4})])
def test_fast_search_candidate_budget(client,path,extras):
    a=[[0.]*20 for _ in range(20)]
    r=client.post(path,json={'row_payoffs':a,'col_payoffs':a,**extras})
    assert r.status_code==422 and 'budget' in r.json()['detail']


def test_exhaustive_pair_budget_is_bounded(client):
    a=[[0.]*9 for _ in range(9)]
    r=client.post('/api/equilibria/hybrid',json={'row_payoffs':a,'col_payoffs':a,'fast_support':1,'exhaustive':True})
    assert r.status_code==422 and 'budget' in r.json()['detail']


@pytest.mark.parametrize('path,symbol,error',[
 ('/api/equilibria/support-lp','solve_support_lp',ValueError('LP failure')),
 ('/api/equilibria/hybrid','solve_hybrid',ValueError('unsupported budget')),
 ('/api/equilibria/lemke-howson','solve_lemke_howson',RuntimeError('pivot unavailable')),
 ('/api/equilibria/independent-lemke-howson','solve_independent_lh',ArithmeticError('unbounded pivot'))])
def test_solver_exceptions_become_http_422(client,monkeypatch,path,symbol,error):
    def raise_error(*a,**kw):raise error
    monkeypatch.setattr(main,symbol,raise_error)
    r=client.post(path,json={'row_payoffs':[[1,-1],[-1,1]],'col_payoffs':[[-1,1],[1,-1]]})
    assert r.status_code==422 and str(error) in r.json()['detail']


def test_api_analytic_wrong_shape_rejected(client):
    r=client.post('/api/equilibria/analytic-2x2',json={'row_payoffs':[[1,2,3],[4,5,6]],'col_payoffs':[[1,2,3],[4,5,6]]})
    assert r.status_code==422


def test_api_bayesian_and_cfr_plus_route_results(client):
    b=client.post('/api/bayesian/evaluate',json={'prior':[1.], 'row_payoffs':[[[1]]], 'col_payoffs':[[[2]]], 'row_strategy':[0],'col_strategy':[0]})
    assert b.status_code==200 and b.json()['expected_payoffs']==[1.,2.]
    c=client.post('/api/imperfect-information/cfr-plus',json={'nodes':{'root':{'payoff':3}},'iterations':1})
    assert c.status_code==200 and c.json()['exploitability']==0


def test_nonsquare_3x3_analyze_has_no_2x2_mixed_claim(client):
    a=[[0,1,0],[1,0,1],[0,1,0]]
    r=client.post('/api/analyze',json={'row_payoffs':a,'col_payoffs':a})
    assert r.status_code==200 and r.json()['interior_mixed_nash_2x2']==[]


def test_direct_zero_sum_rejects_nonfinite_matrix():
    with pytest.raises(HTTPException,match='Invalid matrix'):
        main.zero_sum(main.ZeroSum(payoffs=[[float('nan')]]))


def test_experiment_roundtrip_in_temp_database(tmp_path,monkeypatch):
    monkeypatch.setenv('GAME_LAB_DB',str(tmp_path/'local.sqlite3'))
    saved=research.save_experiment(research.Experiment(title='example',game={'A':[[0]]},notes='note'))
    assert research.get_experiment(saved['id'])['game']=={'A':[[0]]}
    assert research.list_experiments()[0]['id']==saved['id']
    with pytest.raises(HTTPException,match='not found'):research.get_experiment('nonexistent')


def test_experiment_rejects_nan_and_large_record(tmp_path,monkeypatch):
    monkeypatch.setenv('GAME_LAB_DB',str(tmp_path/'local.sqlite3'))
    with pytest.raises(HTTPException,match='finite and JSON serializable'):
        research.save_experiment(research.Experiment(title='bad',game={'x':float('nan')}))
    with pytest.raises(HTTPException,match='storage size budget'):
        research.save_experiment(research.Experiment(title='too large',game={'x':'a'*200001}))


def test_experiment_count_quota(tmp_path,monkeypatch):
    monkeypatch.setenv('GAME_LAB_DB',str(tmp_path/'local.sqlite3'))
    db=research.db_connection()
    try:
        db.executemany('INSERT INTO experiments VALUES (?,?,?,?)',((f'id{i}','item','{}','') for i in range(1000)))
        db.commit()
    finally:db.close()
    with pytest.raises(HTTPException,match='storage quota exceeded'):
        research.save_experiment(research.Experiment(title='new',game={}))


def test_experiment_approximate_byte_quota(tmp_path,monkeypatch):
    monkeypatch.setenv('GAME_LAB_DB',str(tmp_path/'local.sqlite3'))
    db=research.db_connection()
    try:
        db.execute('INSERT INTO experiments VALUES (?,?,?,?)',('id','t','x'*9999990,''));db.commit()
    finally:db.close()
    with pytest.raises(HTTPException,match='storage quota exceeded'):
        research.save_experiment(research.Experiment(title='next',game={'x':'a'*100}))


def test_best_response_rejects_cyclic_information_set_dependencies():
    nodes={'root':{'player':'chance','actions':{'L':'la','R':'rb'},'probabilities':{'L':.5,'R':.5}},
        'la':{'player':0,'infoset':'A','actions':{'go':'lb'}},
        'lb':{'player':0,'infoset':'B','actions':{'go':'t1'}},
        'rb':{'player':0,'infoset':'B','actions':{'go':'ra'}},
        'ra':{'player':0,'infoset':'A','actions':{'go':'t2'}},
        't1':{'payoff':1},'t2':{'payoff':0}}
    with pytest.raises(ValueError,match='Unable to order information sets'):
        best_response(nodes,'root',{},0)
    with pytest.raises(HTTPException,match='Perfect recall'):
        validate_cfr_game(CFRPlusRequest(nodes=nodes,iterations=1))


def test_support_lp_checks_independent_gaps_when_external_lp_lies(monkeypatch):
    def false_success(*a,**kw):
        k=kw['A_eq'].shape[1]-1
        return SimpleNamespace(success=True,x=np.r_[np.full(k,1/k),1/k])
    monkeypatch.setattr(support_lp,'linprog',false_success)
    a=np.array([[1,-1],[-1,1.]])
    r=support_lp.solve_support_lp(a,-a)
    assert r['equilibria']
    for e in r['equilibria']:
        x=np.array(e['row_probabilities']);y=np.array(e['col_probabilities'])
        assert np.max(a@y)-x@a@y<1e-8
        assert np.max(-x@a)-(-x@a)@y<1e-8


def test_coarse_support_deduplication_on_degenerate_game():
    a=np.zeros((2,2))
    fine=support_lp.solve_support_lp(a,a,tol=1e-8)
    coarse=support_lp.solve_support_lp(a,a,tol=.6)
    assert len(fine['equilibria'])==9 and len(coarse['equilibria'])==4
    assert all(e['nash_gap']==0 for e in coarse['equilibria'])
