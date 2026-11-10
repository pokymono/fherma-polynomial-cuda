"""Alternating local timing checks for already exact-validated candidates."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

root=Path(__file__).resolve().parents[1]
candidates={
    'baseline':(root/'build/solution',root/'src/solve.cu'),
    'register_ntt':(root/'build/v022-register-ntt',root/'experiments/v022-register-ntt/solve.cu'),
}
report={'official':False,'hardware':'RTX 3050 Laptop SM86','candidates':{},'runs':[]}
for name,(binary,source) in candidates.items():
    report['candidates'][name]={
        'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    }
for name in ['baseline','register_ntt','register_ntt','baseline']:
    for directory in ['fresh_n32768_w868','fresh_n65536_w1747']:
        path=root/'tests/data'/directory
        result=path/'out/results.json'
        result.unlink(missing_ok=True)
        subprocess.run([str(candidates[name][0]),str(path)],check=True,timeout=180)
        rows=json.loads(result.read_text())['cases']
        expected=json.loads((path/'manifest.json').read_text())['cases']
        if len(rows)!=expected or any(r.get('status')!='ok' or r.get('seconds') is None for r in rows):
            raise RuntimeError(f'{name}: incomplete or crashed harness results')
        # Exclude only the first local call for both versions; this is a proxy
        # timing comparison, not the platform's warmup or ranking procedure.
        times=[r['seconds']*1000 for r in rows[1:]]
        row={'candidate':name,'directory':directory,'median_ms':statistics.median(times),
             'min_ms':min(times),'max_ms':max(times),'timings_ms':times}
        report['runs'].append(row)
        (root/'evidence/nov10_alternating.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='timings_ms'}),flush=True)
