import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAPERS = {
    '1706.03762v7': 'Attention Is All You Need',
    '2410.00037v1': 'Moshi: a speech-text foundation model for real-time dialogue',
    '2507.07955v2': 'Dynamic Chunking for End-to-End Hierarchical Sequence Modeling',
}
SETTINGS = {'small': (120, 2, False), 'balanced': (240, 4, True), 'large': (420, 6, True)}
STOP = set('a an the of in to and is are was were for on by with does do what which how from paper according its it this that as at'.split())


def words(text):
    return [w[:-1] if len(w) > 4 and w.endswith('s') else w for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in STOP]


def focused_query(query):
    """Remove known paper names so repeated titles do not dominate lexical evidence."""
    selected = None
    aliases = {'1706.03762v7': ['attention is all you need', 'transformer'],
               '2410.00037v1': ['moshi'],
               '2507.07955v2': ['dynamic chunking for end-to-end hierarchical sequence modeling', 'dynamic chunking', 'h-net']}
    lowered = query.lower()
    for paper, names in aliases.items():
        if any(name in lowered for name in names):
            # Multi-paper comparison stays unfiltered.
            if selected and selected != paper:
                return query, None
            selected = paper
    for title in PAPERS.values():
        query = re.sub(re.escape(title), ' ', query, flags=re.I)
    for name in ['attention is all you need', 'dynamic chunking', 'moshi', 'h-net']:
        query = re.sub(re.escape(name), ' ', query, flags=re.I)
    return query, selected


def normalize(text):
    return ' '.join(text.split())


def fetch():
    directory = ROOT / 'data/papers'
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for paper, title in PAPERS.items():
            path = directory / f'{paper}.pdf'
            url = f'https://arxiv.org/pdf/{paper}'
            if not path.exists():
                response = client.get(url)
                response.raise_for_status()
                if not response.content.startswith(b'%PDF'):
                    raise ValueError(f'Not a PDF: {url}')
                path.write_bytes(response.content)
                time.sleep(3)  # Be considerate to arXiv; no parallel downloads.
            reader = PdfReader(path)
            pages = [{'paper': paper, 'page': i + 1, 'text': normalize(p.extract_text() or '')}
                     for i, p in enumerate(reader.pages)]
            (directory / f'{paper}.json').write_text(json.dumps(pages, ensure_ascii=False), encoding='utf-8')
            manifest.append({'paper': paper, 'title': title, 'url': url, 'pages': len(pages),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


@dataclass
class Chunk:
    id: str
    paper: str
    page: int
    text: str


class Index:
    def __init__(self, setting='balanced', pages=None):
        size, self.k, self.rerank = SETTINGS[setting]
        if pages is None:
            pages = []
            for paper in PAPERS:
                path = ROOT / f'data/papers/{paper}.json'
                if not path.exists():
                    raise FileNotFoundError('Run python -m relay fetch first')
                pages.extend(json.loads(path.read_text(encoding='utf-8')))
        self.chunks = []
        for page in pages:
            tokens = page['text'].split()
            for start in range(0, len(tokens), size - size // 5):
                self.chunks.append(Chunk(f"{page['paper']}:p{page['page']}:w{start}",
                                         page['paper'], page['page'], ' '.join(tokens[start:start + size])))
        self.counts = [Counter(words(c.text)) for c in self.chunks]
        self.avg = sum(map(lambda c: sum(c.values()), self.counts)) / max(1, len(self.counts))
        df = Counter(w for c in self.counts for w in c)
        self.idf = {w: math.log(1 + (len(self.chunks) - n + .5) / (n + .5)) for w, n in df.items()}

    def search(self, query, k=None):
        query, selected = focused_query(query)
        q = words(query)
        scores = []
        for chunk, count in zip(self.chunks, self.counts):
            if selected and chunk.paper != selected:
                continue
            length = sum(count.values())
            score = sum(self.idf.get(w, 0) * count[w] * 2.5 /
                        (count[w] + 1.5 * (.25 + .75 * length / max(1, self.avg))) for w in q)
            scores.append((score, chunk))
        candidates = sorted(scores, key=lambda x: x[0], reverse=True)[:max(self.k * 4, 12)]
        if self.rerank:
            # Cheap lexical phrase re-ranking; no hidden embedding/model calls.
            pairs = list(zip(q, q[1:]))
            candidates = sorted(candidates, key=lambda x: x[0] +
                                2 * sum(f'{a} {b}' in x[1].text.lower() for a, b in pairs), reverse=True)
        return [asdict(c) for score, c in candidates[:k or self.k] if score > 0]


def extractive_draft(payload):
    """Stand-in semantics: lexical extractive baseline, never reads gold answers."""
    focused, _ = focused_query(payload['question'])
    query = set(words(focused)) - {'many', 'report', 'model', 'used', 'use', 'name', 'call'}
    ranked = []
    for chunk in payload['chunks']:
        for sentence in re.split(r'(?<=[.!?])\s+', chunk['text']):
            overlap = len(query.intersection(words(sentence)))
            if 'how many' in payload['question'].lower() and not re.search(r'\d|\bsix\b|\beight\b', sentence):
                continue
            if 25 <= len(sentence) <= 750:
                ranked.append((overlap / max(1, len(query)) - len(sentence) / 15000, sentence, chunk['id']))
    ranked.sort(reverse=True)
    if not ranked or ranked[0][0] < .44:
        return {'claims': [], 'refuse': True}
    return {'claims': [{'quote': ranked[0][1], 'chunk_id': ranked[0][2]}], 'refuse': False}


def validate_claims(draft, chunks):
    if not isinstance(draft, dict) or draft.get('refuse') is True:
        return []
    lookup = {c['id']: c for c in chunks}
    claims = draft.get('claims')
    if not isinstance(claims, list) or not 1 <= len(claims) <= 3:
        return []
    checked = []
    for claim in claims:
        if not isinstance(claim, dict):
            return []
        quote, cid = claim.get('quote'), claim.get('chunk_id')
        if not isinstance(quote, str) or not isinstance(cid, str):
            return []
        chunk = lookup.get(cid)
        if not chunk or len(quote.strip()) < 20 or normalize(quote) not in normalize(chunk['text']):
            return []
        checked.append((normalize(quote), chunk))
    return checked
