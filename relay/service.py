"""Controllable HTTP/SSE model service. Loopback only by default."""
import asyncio
import hashlib
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .library import extractive_draft


@dataclass
class ServiceConfig:
    rpm: int = 240
    tpm: int = 240000
    rpd: int = 10000
    tpd: int = 10000000
    window_s: float = 6.0  # Compressed quota window; rate scaled by window/60.
    median_ms: float = 65
    sigma: float = .65
    token_ms: float = .6
    failure_rate: float = .025
    slow_rate: float = .06
    slow_ms: float = 850
    load_ms: float = 22
    seed: int = 47
    calibration_source: str = 'synthetic'


def tokens(text):
    return max(1, math.ceil(len(text) / 4))


def create_app(config=None):
    cfg = config or ServiceConfig()
    app = FastAPI(title='Relay stand-in')
    recent = deque()
    state = {'daily_calls': 0, 'daily_tokens': 0, 'active': 0, 'day': int(time.time() // 86400)}
    lock = asyncio.Lock()

    @app.get('/health')
    async def health():
        return {'ok': True, 'calibration': 'synthetic unless calibration.json supplied'}

    @app.post('/openai/v1/chat/completions')
    async def complete(request: Request):
        body = await request.json()
        message = body['messages'][-1]['content']
        payload = json.loads(message)
        stage = payload['stage']
        if stage == 'plan':
            result = {'query': payload['question']}
        elif stage == 'draft':
            result = extractive_draft(payload)
        else:
            result = {'supported': bool(payload.get('claims'))}
        output = json.dumps(result)
        tin, tout = tokens(json.dumps(body['messages'])), tokens(output)
        now = time.monotonic()
        async with lock:
            day = int(time.time() // 86400)
            if day != state['day']:
                state.update(day=day, daily_calls=0, daily_tokens=0)
            while recent and now - recent[0][0] >= cfg.window_s:
                recent.popleft()
            headers = {'x-ratelimit-limit-requests': str(cfg.rpd),
                       'x-ratelimit-limit-tokens': str(cfg.tpm),
                       'x-ratelimit-remaining-requests': str(max(0, cfg.rpd-state['daily_calls'])),
                       'x-ratelimit-remaining-tokens': str(max(0, int(cfg.tpm * cfg.window_s / 60)-sum(x[1] for x in recent)))}
            daily = state['daily_calls'] >= cfg.rpd or state['daily_tokens'] + tin + tout > cfg.tpd
            minute = len(recent) >= max(1, int(cfg.rpm * cfg.window_s / 60)) or \
                sum(x[1] for x in recent) + tin + tout > cfg.tpm * cfg.window_s / 60
            if daily or minute:
                wait = (86400 - time.time() % 86400) if daily else max(.01, cfg.window_s-(now-recent[0][0])) if recent else cfg.window_s
                return JSONResponse({'error': {'code': 'daily_limit' if daily else 'rate_limit',
                                               'message': 'daily quota exhausted' if daily else 'minute quota exhausted'}},
                                    status_code=429, headers={**headers, 'retry-after': str(round(wait, 3))})
            recent.append((now, tin + tout))
            state['daily_calls'] += 1
            state['daily_tokens'] += tin + tout
            state['active'] += 1
            active = state['active']
        identity = request.headers.get('x-request-id', '').replace('retry stress on', 'retry stress off')
        rng = random.Random(int(hashlib.sha256(f'{cfg.seed}:{identity}:{stage}'.encode()).hexdigest()[:16], 16))
        # Fault decision is stable per request+stage; retries can amplify repeated faults.
        failure = rng.random() < cfg.failure_rate
        delay = rng.lognormvariate(math.log(max(1, cfg.median_ms)), cfg.sigma) + cfg.load_ms * max(0, active - 2)
        if rng.random() < cfg.slow_rate:
            delay += cfg.slow_ms
        if failure:
            state['active'] -= 1
            return JSONResponse({'error': {'code': 'server_error', 'message': 'injected failure'}}, status_code=503)

        async def stream():
            try:
                await asyncio.sleep(delay / 1000)
                for i in range(0, len(output), 24):
                    yield 'data: ' + json.dumps({'choices': [{'delta': {'content': output[i:i+24]}}]}) + '\n\n'
                    await asyncio.sleep(cfg.token_ms * tokens(output[i:i+24]) / 1000)
                yield 'data: ' + json.dumps({'choices': [], 'usage': {'prompt_tokens': tin, 'completion_tokens': tout}}) + '\n\n'
                yield 'data: [DONE]\n\n'
            finally:
                state['active'] -= 1
        return StreamingResponse(stream(), media_type='text/event-stream', headers=headers)
    return app
