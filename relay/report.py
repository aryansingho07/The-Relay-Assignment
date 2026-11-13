"""Generate a two-page report directly from the raw CSV and event traces."""
import csv
import json
from html import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, NextPageTemplate, PageBreak,
                               PageTemplate, Paragraph, Spacer, Table, TableStyle)

from .library import ROOT, SETTINGS
from .metrics import load, tables


def fmt(value, places=1):
    return 'not measured' if value is None else f'{value:,.{places}f}'


def build():
    rows = load()
    scenarios, quality = tables(rows)
    (ROOT / 'output/summary.json').write_text(json.dumps({'scenarios': scenarios, 'retrieval': quality}, indent=2))
    traces = {t['request_id']: t['events'] for t in map(json.loads, (ROOT / 'output/traces.jsonl').read_text().splitlines())}
    candidates = [r for r in rows if r['status'] == 'ok' and r['scenario'] != 'retrieval quality']
    retried = [r for r in candidates if int(r['retries']) > 0]
    slow = max(retried or candidates, key=lambda r: float(r['total_ms']))
    events = traces[slow['request_id']]
    waiting = sum(e['duration_ms'] for e in events if e['kind'] == 'queue')
    retry = sum(e['duration_ms'] for e in events if e['kind'] == 'retry_wait')
    working = float(slow['total_ms']) - waiting - retry
    # Store decomposition in raw rows as well: every report number is CSV-recomputable.
    for row in rows:
        ev = traces.get(row['request_id'], [])
        row['trace_wait_ms'] = round(sum(e['duration_ms'] for e in ev if e['kind'] == 'queue'), 3)
        row['trace_retry_ms'] = round(sum(e['duration_ms'] for e in ev if e['kind'] == 'retry_wait'), 3)
        row['trace_work_ms'] = round(float(row['total_ms']) - row['trace_wait_ms'] - row['trace_retry_ms'], 3)
    with (ROOT / 'results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    chosen = max(quality, key=lambda s: (quality[s]['correct_pct'] + quality[s]['refused_correctly_pct'], -quality[s]['tokens']))
    has_real = any(r['source'] == 'real' for r in rows)
    pre = ('Hosted runs are labeled in the results tables.' if has_real else
           'Benchmark scope: local stand-in. Hosted Groq measurements and provider calibration are pending.')
    styles = {
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=8.5, leading=10.6, spaceAfter=4.0),
        'head': ParagraphStyle('head', fontName='Helvetica-Bold', fontSize=9.8, leading=11.8, spaceBefore=5, spaceAfter=3, keepWithNext=True),
        'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=18, leading=21, spaceAfter=8),
        'small': ParagraphStyle('small', fontName='Helvetica', fontSize=7, leading=8.7, spaceAfter=4),
        'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=7.0, leading=8.6),
        'warning': ParagraphStyle('warning', fontName='Helvetica-Bold', fontSize=8, leading=10.5, textColor=colors.HexColor('#a13320'), spaceAfter=6),
    }
    story, plain = [], []

    def p(text, style='body'):
        story.append(Paragraph(text, styles[style])); plain.append(text)

    def heading(title):
        p(title, 'head')

    def table(data, widths):
        content = [[Paragraph(escape(str(cell)).replace('\n', '<br/>'), styles['cell']) for cell in row] for row in data]
        item = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
        item.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dce8ed')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f6f7')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor('#467184')),
            ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 3.3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3.3)]))
        story.append(item); story.append(Spacer(1, 4)); plain.extend(' | '.join(map(str, row)) for row in data)

    p('THE RELAY / Python implementation', 'title')
    p(pre, 'warning')
    heading('1. What I built and why')
    p('Model planning, BM25 retrieval tool, model extractive drafting, local quotation/citation verification, then model relevance checking and output. The decisive choice: buffer until verification. Admission bounds concurrency and queue depth; a deadline includes waiting and retries. Citations retain PDF page and arXiv version. The semantic checker remains fallible.')
    heading('2. The real service and my stand-in')
    real_intro = 'Real measurements are appended in the tables; observed headers are in traces.' if has_real else 'Real account limits, latency and recovery were not measured.'
    p(real_intro + ' Groq documents organization-wide quotas and HTTP 429 with Retry-After. Its requests header means daily requests; its tokens header means minute tokens. The HTTP/SSE stand-in has lognormal prefill, token pacing, injected errors/tails, load-sensitive delays and rolling quotas. Defaults are assumptions. The real command fits prefill median/log dispersion only; provider queues and correlated failures remain unknown.')
    table([
        ['Table 1: limit', 'Value found', 'How found', 'Confidence'],
        ['Requests/minute', '30 documented; account not measured', 'Groq free-plan docs, gpt-oss-20b', 'medium'],
        ['Tokens/minute', '8,000 documented; account not measured', 'Same documentation', 'medium'],
        ['Requests/day', '1,000 documented; account not measured', 'Same documentation', 'medium'],
        ['Tokens/day', '200,000 documented; account not measured', 'Same documentation', 'medium'],
        ['In flight at once', 'not measured', 'No account experiment', 'low'],
        ['Limit response', '429 + Retry-After documented; not observed', 'Groq rate-limit documentation', 'medium'],
        ['Recovery time', 'not measured', 'No real 429 observed', 'low'],
    ], [81, 107, 122, 49])
    heading('3. Results')
    burst = [row for row in rows if row['scenario'] == 'burst' and row['source'] == 'stand-in']
    completed = sum(row['status'] == 'ok' for row in burst)
    p(f'The stand-in burst completed {completed} of {len(burst)} requests. Total-time percentiles include '
      'immediate rejections, which can drive the median down as completion falls. First-token percentiles '
      'count requests that emitted verified output. The tables use wall-clock measurements from results.csv.')
    heading('4. Where the time goes')
    first_ms = float(slow['first_token_ms'])
    total_ms = float(slow['total_ms'])
    p(f'<b>{escape(slow["request_id"])}</b> took {fmt(total_ms)} ms: {fmt(waiting)} ms waiting, '
      f'{fmt(working)} ms working and {fmt(retry)} ms in retry waits. The first verified token appeared '
      f'at {fmt(first_ms)} ms, leaving {fmt(total_ms - first_ms)} ms after it. The trace timestamps '
      'the retry waits before output. Working includes the model, network, retrieval and output pacing.')
    heading('5. When a limit is hit')
    p('An admitted request can wait for one retry per stage after a minute-limit 429. The original deadline still applies. A full queue rejects new arrivals. Low token headroom reduces passage count, which can weaken evidence coverage. Daily exhaustion opens a circuit breaker and returns the budget message to queued and new requests.')
    heading('Retrieval: quality versus cost')
    p(f'Table 3 compares page-local pieces and lexical phrase re-ranking. <b>{chosen}</b> had the highest combined score for correct answers and correct refusals, and used the fewest tokens. Load runs used balanced. Under token pressure the pipeline fetches two passages. The 18-question development set guided retrieval changes. The stand-in checks quotation structure and the grading rubric uses keywords and citation pages.')
    heading('How this library will mislead a naive retriever')
    p('Repeated titles: "Which optimizer trains the Transformer?" can retrieve the author block. Adjacent translation results: "What English-to-German BLEU?" can return English-to-French BLEU. Different latency definitions: "Moshi theoretical and practical latency?" can retrieve an ablation\'s theoretical delay while omitting the practical figure. Title stripping helps; neither citation syntax nor lexical overlap proves relevance.')
    heading('6. Retries')
    p('Each stage can retry a transport error, malformed JSON or transient HTTP response once. Authentication and daily limits fail immediately. In the paired stress test, request 0001 hit a repeatable planning fault. Table 2c shows the added wait and the same terminal error. Retries also raised the scenario timeout rate in Table 2b.')
    heading('7. What the person sees')
    p('At 3pm: "The model budget for today is exhausted. I cannot safely finish your answer now. Please try after the provider quota resets; your request has not been left waiting." The stand-in assumes a UTC-midnight daily reset; Groq reset timing was not measured.')
    heading("8. What I'm unsure about, what I left out, what I learned")
    p('Independent lognormal latency is least certain; correlated congestion would make tails optimistic. Deliberately omitted: distributed scheduling and a neural reranker. Lesson: a citation proves where words occur, not that they answer the question. The keyword/page rubric can overcount correctness or miss valid wording; stored answers need manual review.')
    heading('Your feedback on Vaani')
    p('Homepage review by the assistant. The examples of turn-taking and handling silence made the voice product concrete. The term "voice AI harness" and multiple platform domains made the starting point hard to find. I would feature a sample call with the voice, language and latency conditions next to it.')
    p('Sources checked 10 Nov 2026: console.groq.com/docs/rate-limits; console.groq.com/docs/text-chat; www.vaani.dev. Library manifest records the three exact version URLs and SHA-256 digests.', 'small')

    story.extend([NextPageTemplate('tables'), PageBreak()])
    heading('Results tables / recomputable from results.csv')
    p('p50 / p95 / p99 use linear interpolation. First token = first verified visible answer token, not an internal model delta. All time values are milliseconds except duration. Tokens/answer includes observed tokens spent on failures divided by completed answers; stand-in tokens use ceil(characters/4). Cost is the free API bill, excluding local compute. No completed answers means tokens/answer and cost/1,000 are not measured. Offered rate uses the scheduled arrival span; burst arrivals are simultaneous. Quota window and configurations are raw CSV columns.', 'small')
    data = [['Table 2a: scenario / source', 'Sent / duration (s) / offered per min', 'First-token n', 'First token p50 / p95 / p99 (ms)', 'Total p50 / p95 / p99 (ms)']]
    for key, item in scenarios.items():
        name, source = key.split('|')
        data.append([f'{name} / {source}', f'{item["n"]} / {fmt(item["duration_s"])} / {item["rate"]}', item['first_n'],
                     ' / '.join(map(fmt, item['first'])), ' / '.join(map(fmt, item['total']))])
    if not has_real:
        data.append(['Real service', 'not measured', 'not measured', 'not measured', 'not measured'])
    table(data, [167, 145, 62, 205, 197])
    data = [['Table 2b: scenario', 'Succeeded %', 'Rate-limited %', 'Timeout / given up %', 'Error %', 'Retries / request', 'Peak queue', 'Tokens / answer', 'USD / 1,000 answers']]
    for key, item in scenarios.items():
        data.append([key.split('|')[0], fmt(item['ok_pct']), fmt(item['rate_limited_pct']), fmt(item['timeout_pct']), fmt(item['error_pct']), fmt(item['retries'], 2), item['peak_queue'], fmt(item['tokens']), fmt(item['cost'], 2)])
    if not has_real:
        data.append(['Real service'] + ['not measured'] * 8)
    table(data, [145, 64, 69, 79, 49, 71, 55, 105, 139])
    rows_by_id = {row['request_id']: row for row in rows}
    pair = [rows_by_id.get(f'retry stress {mode}-balanced-0001') for mode in ('off', 'on')]
    if all(pair):
        data = [['Table 2c: retry case 0001', 'Total (ms)', 'Retry wait (ms)', 'Retries', 'Outcome']]
        for row in pair:
            data.append([row['scenario'], fmt(float(row['total_ms'])), fmt(row['trace_retry_ms']),
                         row['retries'], row['status']])
        table(data, [230, 125, 125, 170, 126])
    data = [['Table 3: words / k / re-rank', 'Asked', 'Correct + cited %', 'Wrong %', 'Correct refusal %', 'Wrong refusal %', 'Tokens / answer', 'First p50 / p95 (ms)', 'Total p50 / p95 (ms)']]
    for setting, item in quality.items():
        size, k, rank = SETTINGS[setting]
        data.append([f'{setting}: {size} / {k} / {"yes" if rank else "no"}', item['n'], fmt(item['correct_pct']), fmt(item['wrong_pct']), fmt(item['refused_correctly_pct']), fmt(item['refused_wrongly_pct']), fmt(item['tokens']), ' / '.join(map(fmt, item['first'][:2])), ' / '.join(map(fmt, item['total'][:2]))])
    table(data, [145, 36, 72, 51, 77, 72, 81, 123, 119])
    heading('How I worked with the assistant')
    p('The available prompts are recorded in conversation_history.txt. The first selected Python for the implementation. The second overruled the proposed live Groq run and set the scope of the measurements. The third asked for the GitHub submission and a brief-aligned write-up, which prompted a check of the required tables and retry example.', 'small')
    p('During that check, the assistant found that the earlier write-up discussed retry failures only in aggregate. The paired request trace now shows a repeated planning fault directly. The recorded user decisions concern the language, experiment scope and repository. The assistant wrote the implementation and analyses. A mistake caught personally by the candidate and their own reflection remain to be added.', 'small')

    width, height = landscape(A4)
    margin, gap = 32, 18
    column = (width - 2 * margin - gap) / 2
    frame_args = dict(topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
    frames = [Frame(margin, 29, column, height - 62, **frame_args), Frame(margin + column + gap, 29, column, height - 62, **frame_args)]
    full = Frame(margin, 29, width - 2 * margin, height - 62, **frame_args)

    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor('#bbcbd1')); canvas.line(margin, 22, width-margin, 22)
        canvas.setFont('Helvetica', 7); canvas.drawString(margin, 12, 'THE RELAY | measured locally, synthetic service | 10 Nov 2026')
        canvas.drawRightString(width-margin, 12, str(document.page))

    path = ROOT / 'output/relay_report.pdf'
    doc = BaseDocTemplate(str(path), pagesize=(width, height), title='The Relay - Python experiments', author='Candidate review required')
    doc.addPageTemplates([PageTemplate(id='writing', frames=frames, onPage=footer), PageTemplate(id='tables', frames=[full], onPage=footer)])
    doc.build(story)
    from pypdf import PdfReader
    count = len(PdfReader(path).pages)
    if count != 2:
        raise RuntimeError(f'Report must be two pages, rendered {count}; adjust layout')
    (ROOT / 'output/relay_report.txt').write_text('\n\n'.join(plain), encoding='utf-8')
    print(path)
