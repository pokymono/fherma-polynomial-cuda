"""Fresh exact cases focused on the ranked sizes and arithmetic boundaries."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tests'))
from reference import save_point,verify_point,kronecker

report={'official':False,'source_sha256':hashlib.sha256((root/'src/solve.cu').read_bytes()).hexdigest(),
        'hardware':'RTX 3050 Laptop, SM86','points':[]}
for n,w in [(128,4096),(131072,128),(32768,868),(65536,1747)]:
 path=root/'tests/data'/f'fresh_n{n}_w{w}'
 count=20 if n in (32768,65536) else 4
 start=time.monotonic()
 info=save_point(path,n,w,seeds=range(10001,10001+count),edge_cases=False)
 length=((w+31)//32)*4
 q=int.from_bytes((path/'point/q.bin').read_bytes(),'little')
 last=[0]*n;last[-1]=q-1
 ones=[1]*n
 alternating=[q-1 if i%2 else 0 for i in range(n)]
 limb_boundaries=[min(q-1,(1<<(32*(i%((w+31)//32)+1)))-1) for i in range(n)]
 pairs=[(last,last),(alternating,ones),(limb_boundaries,limb_boundaries),([q//2]*n,[q//2+1]*n)]
 for index,(a,b) in enumerate(pairs,start=count):
  case=path/'cases'/f'{index:06d}';case.mkdir(parents=True,exist_ok=True)
  for name,values in [('a',a),('b',b)]:
   (case/f'{name}.bin').write_bytes(b''.join(x.to_bytes(length,'little') for x in values))
  expected=kronecker(a,b,q)
  reference=path/'reference'/f'{index:06d}';reference.mkdir(parents=True,exist_ok=True)
  (reference/'c.bin').write_bytes(b''.join(x.to_bytes(length,'little') for x in expected))
 manifest=json.loads((path/'manifest.json').read_text());manifest['cases']=count+len(pairs)
 (path/'manifest.json').write_text(json.dumps(manifest))
 print(json.dumps({'N':n,'W':w,'generated':manifest['cases']}),flush=True)
 subprocess.run([str(root/'build/solution'),str(path)],check=True,timeout=300)
 checks=verify_point(path)
 timings=json.loads((path/'out/results.json').read_text())
 info.update(passed=len(checks),cases=len(checks),random_seeds=list(range(10001,10001+count)),timings=timings,
             median_random_ms=statistics.median(x['seconds'] for x in timings['cases'][1:count])*1000,
             validation_wall_seconds=time.monotonic()-start)
 report['points'].append(info)
 (root/'evidence/stress_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'N':n,'W':w,'passed':len(checks),'random_median_ms':info['median_random_ms']}),flush=True)
print('Fresh exact cases passed:',sum(p['passed'] for p in report['points']),flush=True)
