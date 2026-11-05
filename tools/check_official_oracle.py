"""Cross-check a large local case with the platform's published oracle functions."""
import ast
import hashlib
import json
from pathlib import Path
import time

root=Path(__file__).resolve().parents[1]
source=(root/'references/official_bundle/oracle.py').read_text()
parsed=ast.parse(source)
module=ast.Module(body=[x for x in parsed.body if isinstance(x,ast.FunctionDef) and x.name.startswith('_')],type_ignores=[])
namespace={}
exec(compile(module,'public_fherma_oracle','exec'),namespace)
report={'official_run':False,'source':'Published FHERMA oracle functions, local invocation',
        'oracle_sha256':hashlib.sha256(source.encode()).hexdigest(),'points':[]}
for n,w in [(32768,868),(65536,1747)]:
 path=root/'tests/data'/f'n{n}_w{w}'
 length=((w+31)//32)*4
 q=int.from_bytes((path/'point/q.bin').read_bytes(),'little')
 inputs=[]
 for name in ('a','b'):
  data=(path/'cases/000004'/f'{name}.bin').read_bytes()
  inputs.append([int.from_bytes(data[i:i+length],'little') for i in range(0,len(data),length)])
 start=time.monotonic()
 result=namespace['_negacyclic'](*inputs,q)
 raw=b''.join(x.to_bytes(length,'little') for x in result)
 assert raw==(path/'reference/000004/c.bin').read_bytes()
 assert raw==(path/'out/000004/c.bin').read_bytes()
 info={'N':n,'W':w,'case':4,'exact_match':True,'oracle_seconds':time.monotonic()-start}
 report['points'].append(info)
 (root/'evidence/public_oracle_crosscheck.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(info),flush=True)
