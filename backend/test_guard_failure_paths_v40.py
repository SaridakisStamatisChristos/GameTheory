"""ASGI-level negative tests: payload framing, rate/source limits, and compute concurrency."""
import asyncio
from collections import deque
import pytest
from api_guard import LabGuard


def send_req(guard, *, path='/api/analyze', peer='testclient', host='testserver',headers=(),events=None):
    sent=[]
    events=iter(events if events is not None else [{'type':'http.request','body':b'','more_body':False}])
    async def recv():return next(events)
    async def emit(m):sent.append(m)
    scope={'type':'http','path':path,'client':(peer,123),'headers':[(b'host',host.encode()),*headers]}
    asyncio.run(guard(scope,recv,emit))
    codes=[m['status'] for m in sent if m['type']=='http.response.start']
    return codes[0] if codes else None


def make_guard(echo=False,fail=False):
    async def app(scope,receive,send):
        if fail:raise RuntimeError('induced downstream failure')
        msg=await receive()
        assert msg['type']=='http.request'
        if echo:
            assert msg['body']==b'hello world'
            assert (await receive())['type']=='http.disconnect'
        await send({'type':'http.response.start','status':200,'headers':[]})
        await send({'type':'http.response.body','body':b'OK'})
    return LabGuard(app)


def test_websocket_is_not_processed_by_http_guard():
    observed=[]
    async def app(scope,receive,send):observed.append(scope['type'])
    async def unused():return {'type':'websocket.connect'}
    async def emit(m):pass
    asyncio.run(LabGuard(app)({'type':'websocket','headers':[]},unused,emit))
    assert observed==['websocket']


@pytest.mark.parametrize('host',['localhost:8000','127.0.0.1:8000','[::1]:8000','testserver'])
def test_local_http_hosts(host):assert send_req(make_guard(),host=host)==200


def test_remote_auth_and_health(monkeypatch):
    monkeypatch.delenv('GAME_LAB_API_TOKEN',raising=False)
    guard=make_guard()
    assert send_req(guard,peer='203.0.113.9',host='public.example',path='/api/health')==200
    assert send_req(guard,peer='203.0.113.9',host='public.example')==403
    monkeypatch.setenv('GAME_LAB_API_TOKEN','local-only-test-token')
    peer=dict(peer='203.0.113.9',host='public.example',path='/api/experiments')
    assert send_req(guard,**peer,headers=[(b'authorization',b'Bearer bad')])==403
    assert send_req(guard,**peer,headers=[(b'authorization',b'Bearer local-only-test-token')])==200
    assert send_req(guard,path='/api/experiments')==401


def test_invalid_content_length_and_chunked_body_overflow():
    assert send_req(make_guard(),headers=[(b'content-length',b'bad')])==400
    events=[{'type':'http.request','body':b'a'*150000,'more_body':True},
            {'type':'http.request','body':b'b'*150000,'more_body':False}]
    assert send_req(make_guard(),events=events)==413


def test_request_disconnect_and_reassembled_body_once():
    assert send_req(make_guard(),events=[{'type':'http.disconnect'}]) is None
    events=[{'type':'http.request','body':b'hello ','more_body':True},
            {'type':'http.request','body':b'world','more_body':False}]
    assert send_req(make_guard(echo=True),events=events)==200


def test_solver_semaphore_guard_release_and_non_solver_path():
    guard=make_guard()
    assert guard.compute_slots.acquire(blocking=False)
    assert guard.compute_slots.acquire(blocking=False)
    assert send_req(guard,path='/api/zero-sum')==429
    assert send_req(guard,path='/api/experiments')==200
    guard.compute_slots.release()
    assert send_req(guard,path='/api/zero-sum')==200
    assert guard.compute_slots.acquire(blocking=False)
    guard.compute_slots.release();guard.compute_slots.release()


def test_solver_semaphore_released_on_downstream_exception():
    guard=make_guard(fail=True)
    with pytest.raises(RuntimeError,match='induced'):send_req(guard,path='/api/equilibria')
    assert guard.compute_slots.acquire(blocking=False)
    guard.compute_slots.release()


def test_source_budget_denies_unseen_peer_when_full(monkeypatch):
    monkeypatch.setattr('api_guard.time.monotonic',lambda:1000.)
    guard=make_guard()
    for k in range(1024):guard.hits[f'peer-{k}']=deque([1000.])
    assert send_req(guard)==429
    assert len(guard.hits)==1024


def test_source_budget_eviction_and_overfull_legacy_state_recovery(monkeypatch):
    monkeypatch.setattr('api_guard.time.monotonic',lambda:1000.)
    guard=make_guard()
    for k in range(1024):guard.hits[f'peer-{k}']=deque([0.])
    assert send_req(guard)==200
    assert list(guard.hits)==['testclient']
    for k in range(1024):guard.hits[f'peer-{k}']=deque([0.])
    assert len(guard.hits)==1025
    assert send_req(guard)==200
    assert list(guard.hits)==['testclient']


def test_rate_limit_and_expired_window(monkeypatch):
    monkeypatch.setattr('api_guard.time.monotonic',lambda:200.)
    guard=make_guard();guard.hits['testclient']=deque([200.]*1000)
    assert send_req(guard)==429
    guard.hits['testclient']=deque([100.]*1000)
    assert send_req(guard)==200
    assert len(guard.hits['testclient'])==1
