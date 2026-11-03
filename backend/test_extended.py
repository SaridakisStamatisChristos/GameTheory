from fastapi.testclient import TestClient
from main import app
client=TestClient(app)

def test_tree():
    r=client.post('/api/tree/backward-induction',json={'root':'r','nodes':[{'id':'r','player':0,'children':['a','b']},{'id':'a','payoffs':[2,1]},{'id':'b','player':1,'children':['c','d']},{'id':'c','payoffs':[3,0]},{'id':'d','payoffs':[1,4]}]})
    assert r.status_code==200,r.text
    assert r.json()['path']==['r','a']

def test_cycle_rejected():
    r=client.post('/api/tree/backward-induction',json={'root':'a','nodes':[{'id':'a','player':0,'children':['b']},{'id':'b','player':1,'children':['a']}]})
    assert r.status_code==422

def test_repeated():
    r=client.post('/api/repeated/simulate',json={'row_payoffs':[[3,0],[5,1]],'col_payoffs':[[3,5],[0,1]],'row_strategy':[0],'col_strategy':[0],'rounds':3,'discount':0.5})
    assert r.status_code==200,r.text
    assert abs(r.json()['discounted_payoffs'][0]-5.25)<1e-9
