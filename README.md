# The Relay

The Relay is a Python research assistant that retrieves passages from three pinned arXiv papers and verifies quotations and citations before streaming an answer. It has a Groq adapter and a local HTTP/SSE stand-in with configurable latency, failures and quotas.

## Assignment files

| File | Contents |
| --- | --- |
| [Two-page report](output/relay_report.pdf) | Design, quota policy, required result tables, a slow request trace and a paired retry example |
| [results.csv](results.csv) | 302 measured stand-in requests, including answers, outcomes, token estimates and grades |
| [Stage traces](output/traces.jsonl) | Model calls, HTTP statuses, retries, retrieval IDs and timing |
| [Summary](output/summary.json) and [run configuration](output/run_manifest.json) | Aggregates and simulator settings |
| [Questions and rubric](data/questions.json) | 18 development questions, expected answers and citation pages |
| [Assistant prompts](conversation_history.txt) | Available prompts used for the assignment |
| [Remaining requirements](CANDIDATE_REVIEW.txt) | Live provider evidence, calibration and candidate reflection |

## Quick start

Use Python 3.11 or later. From PowerShell, run:

```powershell
git clone https://github.com/aryansingho07/The-Relay-Assignment.git
cd The-Relay-Assignment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m relay fetch
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS or Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. The fetch command downloads the pinned PDFs and extracts page text into `data/papers/`. The [paper manifest](data/papers/manifest.json) records the versions and SHA-256 digests used in the published run.

Start the stand-in in one terminal, then ask a question from another terminal in the same directory.

```powershell
.\.venv\Scripts\python.exe -m relay serve
.\.venv\Scripts\python.exe -m relay ask "How many warmup steps are used in Attention Is All You Need?"
```

## Pipeline

1. An async admission gate caps concurrency and queue depth.
2. The model plans a retrieval query. BM25 searches the pinned papers, optionally followed by lexical reranking.
3. The model proposes short exact quotations and chunk IDs. The pipeline refuses when evidence is insufficient.
4. Local validation checks that each quotation occurs in the cited chunk. Citations identify the paper version and physical PDF page.
5. A model check reviews the draft before the CLI streams the verified output.

Each stage can retry a transient failure once within the request deadline. Minute-limit retries honor the provider's wait interval. Low token headroom reduces passage count, and daily exhaustion returns a budget message. The stand-in uses an extractive baseline and a structural check, so the answer grades measure the local pipeline and its retrieval behavior.

## Reproduce the experiments

Run these commands with the virtual environment active or use its Python executable as above.

```text
python -m relay benchmark
python -m relay report
```

The benchmark runs steady arrivals, a simultaneous burst, sustained overload, a halved minute limit, daily quota exhaustion and paired retry stress. It also asks the same 18 questions at three retrieval settings. The benchmark replaces the current raw CSV and traces. Preserve the published files if you want to compare a new run.

The report uses verified time to first token, total request time and outcome rates. Total time includes queue rejections. First-token samples count only requests that emit verified output. The benchmark uses a compressed quota window and estimates tokens from character counts. The keyword and citation-page rubric measures development-set performance. Machine scheduling and local HTTP timing affect the results.

## Live Groq probe

Live Groq measurements and calibration are pending. To collect them, copy `.env.example` to `.env`, set `GROQ_API_KEY` from your free Groq account and confirm the model and limits in your console. Git ignores the credential file.

```text
python -m relay real --requests 6 --interval 20
python -m relay report
```

The probe appends request rows and records response headers and errors in the stage trace. Successful streamed calls can supply a prefill latency fit for the stand-in. The report distinguishes published quota defaults from measured account behavior. The current PDF reflects the available stand-in data and records the candidate reflection still needed.

## Sources

- [Attention Is All You Need v7](https://arxiv.org/abs/1706.03762v7)
- [Moshi v1](https://arxiv.org/abs/2410.00037v1)
- [Dynamic Chunking v2](https://arxiv.org/abs/2507.07955v2)
- [Groq rate limits](https://console.groq.com/docs/rate-limits)
- [Vaani homepage](https://www.vaani.dev/)
