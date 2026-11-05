"""Compare already validated executables in alternating order on the local GPU."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tests'))
from reference import verify_point
rows=[]
for tile in [512,2048,1024,1024,512,2048]:
 binary=root/'build'/f'solution-tile{tile}'
 for n,w in [(32768,868),(65536,1747)]:
  path=root/'tests/data'/f'fresh_n{n}_w{w}'
  subprocess.run([str(binary),str(path)],check=True,timeout=120)
  verify_point(path)
  timings=json.loads((path/'out/results.json').read_text())
  values=[c['seconds'] for c in timings['cases'][1:20]]
  row={'tile':tile,'N':n,'W':w,'random_median_ms':statistics.median(values)*1000,
       'random_min_ms':min(values)*1000,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
  rows.append(row);print(json.dumps(row),flush=True)
(root/'evidence/tile_comparison.json').write_text(json.dumps({'official':False,'hardware':'RTX 3050 Laptop, SM86','runs':rows},indent=2)+'\n')
