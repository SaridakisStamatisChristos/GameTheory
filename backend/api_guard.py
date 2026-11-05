"""Deliberately small, auditable local-lab exposure and work-budget guard.

NOT a multi-user identity/authorization system.  Public deployments need a
reverse-proxy identity boundary, per-user quotas, and security assessment.
"""
import hmac
import os
import threading
import time
from collections import defaultdict, deque
from fastapi.responses import JSONResponse

LOCAL_HOSTS={'localhost','127.0.0.1','::1','testclient'}
LOCAL_HTTP_HOSTS={'localhost','127.0.0.1','[::1]','testserver'}

class LabGuard:
    def __init__(self, app):
        self.app=app
        self.lock=threading.Lock()
        self.hits=defaultdict(deque)
        self.compute_slots=threading.BoundedSemaphore(2)

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':
            return await self.app(scope,receive,send)
        headers={k.lower():v for k,v in scope.get('headers',[])}
        client=(scope.get('client') or ('',0))[0]
        raw_host=headers.get(b'host',b'').decode('latin1').lower()
        host=(raw_host.split(']')[0]+']') if raw_host.startswith('[') else raw_host.split(':')[0]
        local=client in LOCAL_HOSTS and host in LOCAL_HTTP_HOSTS
        token=os.getenv('GAME_LAB_API_TOKEN','')
        auth=headers.get(b'authorization',b'').decode('latin1')
        authorized=bool(token) and hmac.compare_digest(auth, 'Bearer '+token)
        path=scope.get('path','')
        if path!='/api/health':
            if not local and not authorized:
                return await JSONResponse({'detail':'Remote requests require GAME_LAB_API_TOKEN'},status_code=403)(scope,receive,send)
            if path.startswith('/api/experiments') and token and not authorized:
                return await JSONResponse({'detail':'Experiment access requires a valid bearer token'},status_code=401)(scope,receive,send)
        # Bounded memory, small per-client request rate; only OS process scope.
        now=time.monotonic()
        source_full=False
        with self.lock:
            if client not in self.hits and len(self.hits)>=1024:
                for ip, times in list(self.hits.items()):
                    if not times or times[-1]<now-60:self.hits.pop(ip,None)
                if len(self.hits)>=1024:source_full=True
            if not source_full:
                dq=self.hits[client]
                while dq and dq[0]<now-60: dq.popleft()
                too_many=len(dq)>=(1000 if local else 120)
                if not too_many: dq.append(now)
            if len(self.hits)>1024:
                for ip, times in list(self.hits.items()):
                    if not times or times[-1]<now-60:self.hits.pop(ip,None)
        if source_full:
            return await JSONResponse({'detail':'Request source budget exceeded'},status_code=429)(scope,receive,send)
        if too_many:
            return await JSONResponse({'detail':'Request rate budget exceeded'},status_code=429)(scope,receive,send)
        maximum=262_144
        try:
            content_length=int(headers.get(b'content-length',b'0'))
        except ValueError:
            return await JSONResponse({'detail':'Invalid content length'},status_code=400)(scope,receive,send)
        if content_length>maximum:
            return await JSONResponse({'detail':'Request body exceeds 256 KiB'},status_code=413)(scope,receive,send)
        # Enforce the same limit without relying on the client's Content-Length.
        chunks=[];count=0
        while True:
            event=await receive()
            if event['type']=='http.disconnect':return
            data=event.get('body',b'')
            count+=len(data)
            if count>maximum:
                return await JSONResponse({'detail':'Request body exceeds 256 KiB'},status_code=413)(scope,receive,send)
            chunks.append(data)
            if not event.get('more_body',False):break
        body=b''.join(chunks);del chunks
        supplied=False
        async def replay():
            nonlocal supplied
            if not supplied:
                supplied=True
                return {'type':'http.request','body':body,'more_body':False}
            return {'type':'http.disconnect'}
        heavy=path.startswith(('/api/equilibria','/api/imperfect-information','/api/zero-sum','/api/correlated-equilibrium','/api/multiplayer','/api/stochastic'))
        if heavy and not self.compute_slots.acquire(blocking=False):
            return await JSONResponse({'detail':'Solver concurrency budget exceeded'},status_code=429)(scope,receive,send)
        try:
            return await self.app(scope,replay,send)
        finally:
            if heavy:self.compute_slots.release()
