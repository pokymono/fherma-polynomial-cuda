"""Build a separate, instrumented copy. Never submit this profiling source."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

root=Path(__file__).resolve().parents[1]
cuda=root.parent/'muse-spark-open/.build-deps/nvcc/nvidia/cu13'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=root/'src/solve.cu')
parser.add_argument('--name',default='current')
args=parser.parse_args()
source=args.source.read_text()
profile=source.replace('#include <string>','#include <string>\n#include <cstdio>')
profile=profile.replace('    check(cudaMemcpy(s.a,', '    cudaEvent_t ev[8]; for(auto& e:ev) cudaEventCreate(&e); cudaEventRecord(ev[0]);\n    check(cudaMemcpy(s.a,')
profile=profile.replace('    dim3 full(', '    cudaEventRecord(ev[1]);\n    dim3 full(')
profile=profile.replace('    if(s.n<=65536)', '    cudaEventRecord(ev[2]);\n    if(s.n<=65536)')
profile=profile.replace('    if(s.crt_weights) {', '    cudaEventRecord(ev[3]);\n    if(s.crt_weights) {')
profile=profile.replace('        tensor_product<false>', '        cudaEventRecord(ev[4]);\n        tensor_product<false>')
if 'else if(s.gap&&s.wide<=128)' in profile:
    marker='        if(s.direct_crt) {\n            if(s.wide<=32)'
    assert marker in profile
    profile=profile.replace(marker,'        cudaEventRecord(ev[5]);\n'+marker)
else:
    profile=profile.replace('        if(s.gap&&s.wide<=128)', '        cudaEventRecord(ev[5]);\n        if(s.gap&&s.wide<=128)')
profile=profile.replace('    check(cudaGetLastError()', '    cudaEventRecord(ev[6]);\n    check(cudaGetLastError()')
profile=profile.replace('"copy answer"); return out;', '"copy answer"); cudaEventRecord(ev[7]); cudaEventSynchronize(ev[7]); for(int i=0;i<7;++i) {float ms=0; cudaEventElapsedTime(&ms,ev[i],ev[i+1]); std::fprintf(stderr,"%s%.6f",i?",":"profile_ms:",ms);} std::fprintf(stderr,"\\n"); for(auto e:ev) cudaEventDestroy(e); return out;')
path=root/'experiments'/f'{args.name}_profile.cu';path.write_text(profile)
binary=root/'build'/f'{args.name}-profile'
command=[str(cuda/'bin/nvcc'),'-std=c++17','-O3','-arch=sm_86','-I'+str(args.source.resolve().parent),'-I'+str(root/'src'),'-I'+str(root.parent/'muse-spark-open/.build-deps/cccl/nvidia/cu13/include/cccl'),'-cudart=shared','-cudadevrt=none','-L'+str(cuda/'lib'),str(root/'src/main.cpp'),str(path),'-o',str(binary),'-Xlinker','-rpath','-Xlinker',str(cuda/'lib')]
subprocess.run(command,check=True)
r=subprocess.run([str(binary),str(root/'tests/data/n65536_w1747')],capture_output=True,text=True,check=True)
result=json.loads((root/'tests/data/n65536_w1747/out/results.json').read_text())
expected=json.loads((root/'tests/data/n65536_w1747/manifest.json').read_text())['cases']
assert len(result['cases'])==expected and all(c['status']=='ok' and c['seconds'] is not None for c in result['cases']),r.stderr
rows=[[float(x) for x in line.split(':')[1].split(',')] for line in r.stderr.splitlines() if line.startswith('profile_ms:')]
assert len(rows)==expected and all(len(row)==7 and all(x>0 for x in row) for row in rows),r.stderr
report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'official':False,'stages':['h2d','encode','ntt','transpose','crt_mma','crt_reduce','d2h'],'ms':rows}
(root/'evidence'/f'{args.name}_profile.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
