"""Exact minimax endpoint integration, validation, and compatibility."""
from fractions import Fraction as F
from fastapi.testclient import TestClient
from main import app


def test_exact_zero_sum_http_and_legacy_independence():
    game = {'payoffs': [[1, -1], [-1, 1]]}
    with TestClient(app, raise_server_exceptions=False) as client:
        result = client.post('/api/zero-sum/exact', json=game)
        assert result.status_code == 200, result.text
        out = result.json()
        assert out['status'] == 'certified_exact_minimax'
        assert out['value_exact'] == '0'
        assert out['minimax_duality_gap_exact'] == '0'
        assert [F(p) for p in out['row_strategy_exact']] == [F(1, 2)]*2
        assert [F(p) for p in out['col_strategy_exact']] == [F(1, 2)]*2
        assert client.post('/api/zero-sum', json=game).status_code == 200
        assert client.post('/api/zero-sum/exact', json={
            **game, 'max_operations': 1}).status_code == 422
        assert client.post('/api/zero-sum/exact', json={
            'payoffs': [[0]*11 for _ in range(10)]}).status_code == 422
        assert client.post('/api/zero-sum/exact', json={
            'payoffs': [[0]*21]}).status_code == 422
        assert client.post('/api/zero-sum/exact', json={
            'payoffs': [[0], [0, 1]]}).status_code == 422
        assert client.post('/api/zero-sum/exact', json={
            'payoffs': [[0]], 'max_operations': 0}).status_code == 422
