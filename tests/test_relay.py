import asyncio
import json

import httpx

from relay.experiment import grade
from relay.library import Index, validate_claims
from relay.pipeline import ModelClient, Pipeline
from relay.service import ServiceConfig, create_app


PAGES = [{'paper': 'testv1', 'page': 3, 'text': 'The encoder contains six identical layers. The decoder has masked attention.'}]


def test_citation_validation_rejects_invented_quote_and_id():
    chunk = Index('small', pages=PAGES).search('encoder')[0]
    assert validate_claims({'claims': [{'quote': 'The encoder contains six identical layers.', 'chunk_id': chunk['id']}]}, [chunk])
    assert not validate_claims({'claims': [{'quote': 'The encoder contains ten identical layers.', 'chunk_id': chunk['id']}]}, [chunk])
    assert not validate_claims({'claims': [{'quote': chunk['text'], 'chunk_id': 'invented'}]}, [chunk])
    assert not validate_claims({'claims': 'malformed'}, [chunk])


def test_chunk_citations_never_cross_pages():
    pages = PAGES + [{'paper': 'testv1', 'page': 4, 'text': 'Different page with a different statement.'}]
    index = Index('small', pages=pages)
    assert all('Different page' not in c.text for c in index.chunks if c.page == 3)


def test_daily_limit_circuit_breaker_no_retry():
    async def run():
        app = create_app(ServiceConfig(rpd=0))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as http:
            model = ModelClient('http://test', 'stand-in', http)
            pipe = Pipeline(Index('small', pages=PAGES), model)
            first, events = await pipe.run('encoder layers', 'one', 'daily', 'small')
            second, second_events = await pipe.run('encoder layers', 'two', 'daily', 'small')
            assert first['status'] == second['status'] == 'rate_limited'
            assert first['retries'] == 0 and not second_events
            assert model.daily_exhausted
    asyncio.run(run())


def test_minute_and_token_limits_return_retry_after():
    async def run():
        for config in [ServiceConfig(rpm=1), ServiceConfig(tpm=1)]:
            app = create_app(config)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as http:
                body = {'messages': [{'content': json.dumps({'stage': 'plan', 'question': 'hi'})}]}
                if config.rpm == 1:
                    assert (await http.post('/openai/v1/chat/completions', json=body)).status_code == 200
                response = await http.post('/openai/v1/chat/completions', json=body)
                assert response.status_code == 429
                assert float(response.headers['retry-after']) > 0
    asyncio.run(run())


def test_deadline_includes_queue_and_cancelled_model():
    async def run():
        app = create_app(ServiceConfig(median_ms=1000, failure_rate=0, slow_rate=0))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as http:
            pipe = Pipeline(Index('small', pages=PAGES), ModelClient('http://test', 'stand-in', http), concurrency=1, queue_size=1, deadline_s=.02)
            results = await asyncio.gather(*(pipe.run('encoder layers', str(i), 'queue', 'small') for i in range(3)))
            assert [r[0]['status'] for r in results] == ['timeout', 'timeout', 'rate_limited']
            assert pipe.pending == 0 and pipe.active == 0
    asyncio.run(run())


def test_quality_does_not_reward_infrastructure_failure():
    assert grade({'status': 'timeout', 'answer': ''}, {'refuse': True}) == 'refused_wrongly'


def test_percentiles_and_failure_denominators():
    from relay.metrics import percentile, summarize
    assert percentile([], 95) is None
    assert percentile([0, 100], 95) == 95
    base = dict(scenario_duration_s='1', offered_rate_per_min='60', retries='0', peak_queue='0', tokens_in='10', tokens_out='2', cost_usd='0', answer_grade='correct')
    stats = summarize([{**base, 'status': 'ok', 'first_token_ms': '100', 'total_ms': '120'},
                       {**base, 'status': 'timeout', 'first_token_ms': '', 'total_ms': '2500'}])
    assert stats['first_n'] == 1 and stats['first'] == [100, 100, 100]
    assert stats['ok_pct'] == stats['timeout_pct'] == 50
    assert stats['tokens'] == 24  # Observed failed work belongs in the denominator accounting.


def test_json_parse_failure_retries_but_auth_does_not():
    async def run():
        from relay.pipeline import Trace, ModelFailure
        calls = []
        def handler(request):
            calls.append(request)
            text = 'not json' if len(calls) == 1 else '{"query":"encoder"}'
            return httpx.Response(200, text='data: '+json.dumps({'choices':[{'delta':{'content':text}}]})+'\n\ndata: [DONE]\n\n')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url='http://test') as http:
            row = {'request_id':'json', 'retries':0, 'tokens_in':0, 'tokens_out':0, 'tokens_estimated':False}
            result = await ModelClient('http://test','stand-in',http).call('plan', {'question':'encoder'}, row, Trace())
            assert result == {'query':'encoder'} and row['retries'] == 1 and len(calls) == 2
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={'error':'unauthorized'})), base_url='http://test') as http:
            row = {'request_id':'auth', 'retries':0}
            try:
                await ModelClient('http://test','stand-in',http).call('plan', {'question':'encoder'}, row, Trace())
                assert False
            except ModelFailure:
                assert row['retries'] == 0
    asyncio.run(run())
