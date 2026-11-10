"""Verify a rebuilt executable against previously generated independent cases."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'tests'))
from reference import verify_point

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('report')
parser.add_argument('--binary', type=Path, default=root / 'build/solution')
parser.add_argument('--source', type=Path, default=root / 'src/solve.cu')
parser.add_argument('--point', action='append', help='Existing case-directory name; repeatable')
args = parser.parse_args()
binary = args.binary.resolve()
source = args.source.resolve()
report = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'source_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in source.parent.iterdir()
                                  if p.suffix in {'.cu', '.cpp', '.h', '.hpp', '.cuh'} or p.name == 'CMakeLists.txt'},
          'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
          'source': str(source), 'binary': str(binary),
          'hardware': 'RTX 3050 Laptop, SM86', 'official': False, 'points': []}
name = args.report
paths=[p for p in (root / 'tests/data').iterdir() if (p / 'manifest.json').exists()]
if args.point:
    paths = [p for p in paths if p.name in args.point]
    missing = set(args.point) - {p.name for p in paths}
    if missing:
        raise ValueError(f'Unknown case directories: {sorted(missing)}')
for path in sorted(paths, key=lambda p: json.loads((p / 'manifest.json').read_text())['point']['N']):
    if not (path / 'reference').exists():
        continue
    results_path = path / 'out/results.json'
    results_path.unlink(missing_ok=True)
    subprocess.run([str(binary), str(path)], check=True, timeout=300)
    timings = json.loads(results_path.read_text())
    manifest = json.loads((path / 'manifest.json').read_text())
    cases = timings.get('cases', [])
    if len(cases) != manifest['cases'] or not cases:
        raise RuntimeError(f'{path.name}: incomplete harness results')
    failures = [c for c in cases if c.get('status') != 'ok' or c.get('seconds') is None]
    if failures:
        raise RuntimeError(f'{path.name}: harness failures; old outputs are not valid evidence: {failures[:2]}')
    checks = verify_point(path)
    point = json.loads((path / 'manifest.json').read_text())['point']
    report['points'].append(dict(point, directory=path.name, passed=len(checks), timings=timings))
    (root / 'evidence' / name).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(point, passed=len(checks), median_ms=statistics.median(c['seconds'] for c in timings['cases'])*1000)), flush=True)
print('Exact checks passed:', sum(x['passed'] for x in report['points']), flush=True)
