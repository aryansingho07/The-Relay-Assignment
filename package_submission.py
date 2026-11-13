"""Package only approved deliverables; never include .env, caches or temp files."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent


def main():
    names = ['README.md', 'requirements.txt', '.env.example', '.gitignore',
             'conversation_history.txt', 'CANDIDATE_REVIEW.txt', 'results.csv', 'package_submission.py']
    files = [ROOT / name for name in names]
    for directory in ('relay', 'tests'):
        files.extend((ROOT / directory).glob('*.py'))
    files.extend((ROOT / 'data').glob('*.json'))
    files.extend((ROOT / 'data/papers').glob('*.json'))
    files.extend((ROOT / 'data/papers').glob('*.pdf'))
    for name in ['relay_report.pdf', 'relay_report.txt', 'summary.json', 'traces.jsonl', 'run_manifest.json', 'calibration.json', 'calibration_notes.json']:
        path = ROOT / 'output' / name
        if path.exists():
            files.append(path)
    files = sorted(set(files))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    archive = ROOT / 'output/The_Relay_Python.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as output:
        for path in files:
            output.write(path, path.relative_to(ROOT))
        output.writestr('submission_manifest.json', json.dumps(manifest, indent=2))
    with ZipFile(archive) as check:
        assert check.testzip() is None
        assert '.env' not in check.namelist()
        assert not any('__pycache__' in name or name.startswith('tmp/') for name in check.namelist())
    print(f'{archive}: {len(files)} files, {archive.stat().st_size:,} bytes; verified')


if __name__ == '__main__':
    main()
