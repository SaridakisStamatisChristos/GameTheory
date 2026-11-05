"""End-to-end HTTP integration and validation for v57 Shapley solver."""
from fastapi.testclient import TestClient
from main import app


def call(payload):
    with TestClient(app, raise_server_exceptions=False) as client:
        return client.post('/api/stochastic/zero-sum/discounted',json=payload)


def test_discounted_stochastic_http_matching_pennies():
    out=call({'rewards':[[[1,-1],[-1,1]]],
              'transitions':[[[[1],[1]],[[1],[1]]]],
              'discount':.8,'tolerance':1e-6})
    assert out.status_code==200,out.text
    data=out.json()
    assert data['status']=='converged_numerical'
    assert data['value_estimates']==[0]
    assert data['row_stationary_policy'][0]==[.5,.5]
    assert data['col_stationary_policy'][0]==[.5,.5]
    assert data['tolerance_met'] is True


def test_stochastic_http_single_action_and_partial_budget():
    base={'rewards':[[[1]]],'transitions':[[[[1]]]],
          'discount':.95,'tolerance':1e-10,'max_iterations':1}
    p=call(base)
    assert p.status_code==200,p.text
    assert p.json()['status']=='partial_bound'
    assert p.json()['tolerance_met'] is False
    good=call({**base,'discount':0})
    assert good.status_code==200,good.text
    assert good.json()['status']=='converged_numerical'
    assert good.json()['value_estimates']==[1.]


def test_stochastic_http_bad_payloads():
    base={'rewards':[[[1]]],'transitions':[[[[1]]]]}
    for extra in ({'discount':1},{'discount':-1},{'max_iterations':0},
                  {'tolerance':0},{'discount':'NaN'}):
        response=call({**base,**extra})
        assert response.status_code==422,(extra,response.text)
    for bad in ({**base,'transitions':[[[[.5]]]]},
                {**base,'transitions':[[[[-.1]]]]},
                {**base,'rewards':[[[0,1]]]},
                {**base,'rewards':[[[1e7]]]}):
        response=call(bad)
        assert response.status_code==422,(bad,response.text)


def test_stochastic_http_zero_sum_existing_endpoint_unaffected():
    with TestClient(app,raise_server_exceptions=False) as client:
        result=client.post('/api/zero-sum/exact',json={
            'payoffs':[[1,-1],[-1,1]]})
    assert result.status_code==200,result.text
    assert result.json()['value_exact']=='0'
