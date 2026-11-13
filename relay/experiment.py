import asyncio
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import uvicorn

from .library import Index, ROOT, SETTINGS
from .pipeline import ModelClient, Pipeline, REFUSAL
from .service import ServiceConfig, create_app


def grade(row, question):
    # Predeclared lexical/page rubric, deliberately not an LLM judge.
    # Failures are separately reported; refusing due to outage is not a correct abstention.
    if row['status'] != 'ok':
        return 'refused_wrongly'
    refused = row['answer'] == REFUSAL
    if question.get('refuse'):
        return 'refused_correctly' if refused else 'wrong'
    if refused:
        return 'refused_wrongly'
    answer = row['answer'].lower()
    matches = [p in answer for p in question['patterns']]
    content = all(matches) if question.get('all_patterns') else any(matches)
    cited = any(f"arxiv:{question['paper']}, p. {p}]" in answer for p in question['pages'])
    return 'correct' if content and cited else 'wrong'


async def run_scenario(name, n, interval, setting, cfg, source='stand-in', retries=1, deadline=2.5, port=8765):
    server, server_task = None, None
    if source == 'stand-in':
        server = uvicorn.Server(uvicorn.Config(create_app(cfg), host='127.0.0.1', port=port, log_level='error'))
        server_task = asyncio.create_task(server.serve())
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(.02)
        if not server.started:
            raise RuntimeError('Stand-in did not start')
    base = f'http://127.0.0.1:{port}' if source == 'stand-in' else 'https://api.groq.com'
    model = ModelClient(base, source)
    pipeline = Pipeline(Index(setting), model, retries=retries, deadline_s=deadline,
                        near_limit=name == 'per-minute limit halved')
    questions = json.loads((ROOT / 'data/questions.json').read_text())
    started = time.perf_counter()

    async def one(i):
        scheduled = started + i * interval
        await asyncio.sleep(max(0, scheduled - time.perf_counter()))
        question = questions[i % len(questions)]
        row, events = await pipeline.run(question['question'], f'{name}-{setting}-{i:04d}', name, setting)
        row['answer_grade'] = grade(row, question)
        row.update(question_id=question['id'], dispatch_lag_ms=round(max(0, time.perf_counter()-scheduled-row['total_ms']/1000)*1000, 3))
        return row, events

    try:
        results = await asyncio.gather(*(one(i) for i in range(n)))
    finally:
        await model.close()
        if server:
            server.should_exit = True
            await server_task
    duration = time.perf_counter() - started
    rows, traces = [], []
    for row, events in results:
        row.update(scenario_duration_s=round(duration, 6), offered_rate_per_min=round(n * 60 / max(.001, (n-1)*interval), 3) if interval else 'simultaneous',
                   peak_queue=pipeline.peak_queue, cost_usd=0, quota_window_s=cfg.window_s if source == 'stand-in' else 60,
                   configured_rpm=cfg.rpm, configured_tpm=cfg.tpm, configured_rpd=cfg.rpd,
                   configured_tpd=cfg.tpd, calibration=cfg.calibration_source if source == 'stand-in' else 'real',
                   documented_rpm=30, documented_tpm=8000, documented_rpd=1000, documented_tpd=200000)
        rows.append(row)
        traces.append({'request_id': row['request_id'], 'events': events})
    return rows, traces


def save(rows, traces, append=False):
    path = ROOT / 'results.csv'
    if append and path.exists():
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream)) + rows
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / 'output/traces.jsonl').open('a' if append else 'w', encoding='utf-8') as stream:
        for item in traces:
            stream.write(json.dumps(item) + '\n')


async def suite():
    rows, traces = [], []
    calibration = ROOT / 'output/calibration.json'
    base = json.loads(calibration.read_text()) if calibration.exists() else {}
    allowed = set(asdict(ServiceConfig()))
    base = {k: v for k, v in base.items() if k in allowed}
    cases = [
        ('steady', 18, .70, {}, 1),
        ('burst', 60, 0, {}, 1),
        ('sustained overload', 80, .04, {}, 1),
        ('per-minute limit halved', 36, .09, {'rpm': 120}, 1),
        ('daily budget exhausted', 18, .02, {'rpd': 0}, 1),
        ('retry stress off', 18, .08, {'failure_rate': .35, 'slow_rate': .2}, 0),
        ('retry stress on', 18, .08, {'failure_rate': .35, 'slow_rate': .2}, 1),
    ]
    for name, n, interval, overrides, retries in cases:
        cfg = ServiceConfig(**{**base, **overrides})
        r, t = await run_scenario(name, n, interval, 'balanced', cfg, retries=retries)
        rows.extend(r); traces.extend(t)
        print(name, {s: sum(x['status'] == s for x in r) for s in ('ok', 'rate_limited', 'timeout', 'error')}, flush=True)
    for setting in SETTINGS:
        cfg = ServiceConfig(**{**base, 'rpm': 100000, 'tpm': 10000000, 'failure_rate': 0, 'slow_rate': 0})
        r, t = await run_scenario('retrieval quality', 18, .16, setting, cfg, deadline=8)
        rows.extend(r); traces.extend(t)
        print('retrieval', setting, {g: sum(x['answer_grade'] == g for x in r) for g in ('correct', 'wrong', 'refused_correctly', 'refused_wrongly')}, flush=True)
    save(rows, traces)
    (ROOT / 'output/run_manifest.json').write_text(json.dumps({'suite': 'stand-in', 'config': asdict(ServiceConfig(**base)),
        'note': 'Real calibration absent unless output/calibration.json was produced by the real command. Wall-clock measurements, compressed quota windows; tokens estimated as ceil(chars/4).'}, indent=2))
