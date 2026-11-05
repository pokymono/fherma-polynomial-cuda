"""Create only our private FHERMA draft; no benchmark is queued by this script."""
import json
import subprocess
from pathlib import Path
from fherma.api import Api
from fherma.config import load

root=Path(__file__).resolve().parents[1]
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
body={
 'slug':'pokymono-cuda-rns',
 'name':'CUDA RNS with exact integer tensor cores',
 'summary':'Exact negacyclic multiplication using 31-bit RNS transforms and integer tensor-core conversions.',
 'description':'All input conversion, transforms, exact CRT reconstruction, modular reduction, and host/device transfers occur inside the timed run. Parameter-only plans are prepared in init. Local validation has passed; official runner results are pending.',
 'kernel_spec_slug':'negacyclic','kernel_spec_version':'1.0.0',
 'coverage':{'blocks':[{'types':{},'consts':{k:[{'kind':'set','values':[v]}] for k,v in p.items()}} for p in [{'N':65536,'W':1747},{'N':32768,'W':868}]]},
 'mechanism':{'family':'plain','scheme':None,'library':'CUDA'},
 'artifacts':[{'kind':'repository','location':'https://github.com/pokymono/fherma-polynomial-cuda','digest':commit,'path':'src'}],
 'language':'cpp','image':'cupqc','visibility':'private','lifecycle_status':'draft',
 'requires':[],'environment_requirements':[{'name':'gpu'},{'name':'cuda','min':'12.8'}],
}
(root/'evidence/platform_draft_request.json').write_text(json.dumps(body,indent=2)+'\n')
a=Api(load())
try:
 existing=a._request('GET','/kernels/polynomial-multiplication/implementations')
 if any(x.get('slug')==body['slug'] for x in existing.get('items',[])):
  print('Draft already exists; skipping creation.')
 else:
  response=a._client.post('/kernels/polynomial-multiplication/implementations',json=body)
  result=response.json()
  (root/'evidence/platform_draft_response.json').write_text(json.dumps(result,indent=2)+'\n')
  print('HTTP',response.status_code)
  if response.status_code>=400:print(json.dumps(result))
  else: print(json.dumps({k:result.get(k) for k in ['id','slug','name','visibility','lifecycle_status','language','image','runnable']}))
finally:a.close()
