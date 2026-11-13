import argparse
import asyncio
import json
import math
import os
import statistics
import uuid
from dataclasses import asdict

from dotenv import load_dotenv

from .experiment import run_scenario, save, suite
from .library import Index, ROOT, fetch
from .pipeline import ModelClient, Pipeline
from .service import ServiceConfig, create_app


async def ask(args):
    if args.source == 'real' and not os.getenv('GROQ_API_KEY'):
        raise SystemExit('Set GROQ_API_KEY in .env first. Never commit it.')
    model = ModelClient('https://api.groq.com' if args.source == 'real' else args.url, args.source)
    try:
        row, trace = await Pipeline(Index(args.setting), model, deadline_s=args.deadline).run(args.question, 'interactive-' + uuid.uuid4().hex, 'interactive', args.setting, on_token=lambda token: print(token, end='', flush=True))
        if row['status'] != 'ok':
            print(row['answer'])
        else:
            print()
        print(json.dumps({k: row[k] for k in ('status', 'first_token_ms', 'total_ms', 'tokens_in', 'tokens_out')}, indent=2))
    finally:
        await model.close()


async def real(args):
    if not os.getenv('GROQ_API_KEY'):
        raise SystemExit('Real measurements not run: GROQ_API_KEY is absent.')
    # Modest, explicitly bounded; no loops that burn a daily budget looking for 429s.
    rows, traces = await run_scenario('real modest', args.requests, args.interval, 'balanced', ServiceConfig(),
                                     source='real', retries=0, deadline=45)
    save(rows, traces, append=True)
    calls = [e for t in traces for e in t['events'] if e['kind'] == 'model' and e.get('http_status') == 200]
    if calls:
        first = [e['provider_first_token_ms'] - e['start_ms'] for e in calls if e.get('provider_first_token_ms') is not None]
        logs = [math.log(max(1, x)) for x in first]
        cfg = asdict(ServiceConfig())
        if first:
            cfg['median_ms'] = statistics.median(first)
            cfg['sigma'] = statistics.pstdev(logs)
        cfg.update(rpm=30, tpm=8000, rpd=1000, tpd=200000, window_s=60, calibration_source='real-prefill-only')
        (ROOT / 'output/calibration.json').write_text(json.dumps(cfg, indent=2))
        (ROOT / 'output/calibration_notes.json').write_text(json.dumps({
            'successful_calls': len(calls), 'method': 'Fit lognormal prefill median/log sigma from successful real SSE calls. Quotas use documented defaults; inspect observed headers before relying on them.',
            'not_calibrated': ['load_ms', 'slow_rate', 'failure_rate', 'token_ms', 'provider internal queues'],
            'observed_headers': [e.get('headers', {}) for e in calls]}, indent=2))
    print('Saved real rows and response headers. An observed 429 is not guaranteed; report not measured if absent.')


def main():
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description='The Relay research assistant and experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('fetch')
    sub.add_parser('benchmark')
    sub.add_parser('report')
    server = sub.add_parser('serve')
    server.add_argument('--port', type=int, default=8765)
    server.add_argument('--config', help='JSON with ServiceConfig values')
    query = sub.add_parser('ask')
    query.add_argument('question')
    query.add_argument('--source', choices=['stand-in', 'real'], default='stand-in')
    query.add_argument('--url', default='http://127.0.0.1:8765')
    query.add_argument('--setting', choices=['small', 'balanced', 'large'], default='small')
    query.add_argument('--deadline', type=float, default=15)
    probe = sub.add_parser('real')
    probe.add_argument('--requests', type=int, default=6)
    probe.add_argument('--interval', type=float, default=20)
    args = parser.parse_args()
    (ROOT / 'output').mkdir(exist_ok=True)
    if args.command == 'fetch':
        print(json.dumps(fetch(), indent=2))
    elif args.command == 'serve':
        import uvicorn
        cfg = ServiceConfig(**json.loads(open(args.config).read())) if args.config else ServiceConfig()
        uvicorn.run(create_app(cfg), host='127.0.0.1', port=args.port)
    elif args.command == 'benchmark':
        asyncio.run(suite())
    elif args.command == 'report':
        from .report import build
        build()
    elif args.command == 'real':
        if not 1 <= args.requests <= 18:
            parser.error('--requests must be between 1 and 18')
        asyncio.run(real(args))
    else:
        asyncio.run(ask(args))


if __name__ == '__main__':
    main()
