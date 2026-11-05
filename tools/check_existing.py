"""Verify a rebuilt executable against previously generated independent cases."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'tests'))
from reference import verify_point

report = {'source_sha256': hashlib.sha256((root / 'src/solve.cu').read_bytes()).hexdigest(),
          'hardware': 'RTX 3050 Laptop, SM86', 'official': False, 'points': []}
name = sys.argv[1]
paths=[p for p in (root / 'tests/data').iterdir() if (p / 'manifest.json').exists()]
for path in sorted(paths, key=lambda p: json.loads((p / 'manifest.json').read_text())['point']['N']):
    if not (path / 'reference').exists():
        continue
    subprocess.run([str(root / 'build/solution'), str(path)], check=True, timeout=300)
    checks = verify_point(path)
    point = json.loads((path / 'manifest.json').read_text())['point']
    timings = json.loads((path / 'out/results.json').read_text())
    report['points'].append(dict(point, directory=path.name, passed=len(checks), timings=timings))
    (root / 'evidence' / name).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(point, passed=len(checks), median_ms=statistics.median(c['seconds'] for c in timings['cases'])*1000)), flush=True)
print('Exact checks passed:', sum(x['passed'] for x in report['points']), flush=True)
