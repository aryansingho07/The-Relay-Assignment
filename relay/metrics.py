"""All result tables derive from results.csv, not trace files or hidden counters."""
import csv
from collections import defaultdict

from .library import ROOT


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def load():
    with (ROOT / 'results.csv').open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def summarize(rows):
    count = len(rows)
    ok = [r for r in rows if r['status'] == 'ok']
    first = [float(r['first_token_ms']) for r in rows if r['first_token_ms'] != '']
    total = [float(r['total_ms']) for r in rows]
    return {'n': count, 'duration_s': float(rows[0]['scenario_duration_s']), 'rate': rows[0]['offered_rate_per_min'],
            'first': [percentile(first, p) for p in (50, 95, 99)], 'first_n': len(first),
            'total': [percentile(total, p) for p in (50, 95, 99)],
            'ok_pct': 100 * len(ok) / count,
            **{f'{s}_pct': 100 * sum(r['status'] == s for r in rows) / count for s in ('rate_limited', 'timeout', 'error')},
            'retries': sum(int(r['retries']) for r in rows) / count,
            'peak_queue': max(int(r['peak_queue']) for r in rows),
            # Operational tokens/answer includes work spent on failed requests.
            'tokens': sum(int(r['tokens_in']) + int(r['tokens_out']) for r in rows) / len(ok) if ok else None,
            'cost': 1000 * sum(float(r['cost_usd']) for r in rows) / len(ok) if ok else None,
            **{f'{g}_pct': 100 * sum(r['answer_grade'] == g for r in rows) / count for g in ('correct', 'wrong', 'refused_correctly', 'refused_wrongly')}}


def tables(rows):
    grouped = defaultdict(list)
    quality = defaultdict(list)
    for row in rows:
        if row['scenario'] == 'retrieval quality':
            quality[row['retrieval_setting']].append(row)
        else:
            grouped[(row['scenario'], row['source'])].append(row)
    return {f'{name}|{source}': summarize(items) for (name, source), items in grouped.items()}, {name: summarize(items) for name, items in quality.items()}
