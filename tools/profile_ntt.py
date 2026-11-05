"""Build a separate, instrumented copy. Never submit this profiling source."""
from pathlib import Path
import hashlib
import json
import subprocess

root=Path(__file__).resolve().parents[1]
cuda=root.parent/'muse-spark-open/.build-deps/nvcc/nvidia/cu13'
source=(root/'src/solve.cu').read_text()
profile=source.replace('#include <string>','#include <string>\n#include <cstdio>')
profile=profile.replace('    check(cudaMemcpy(s.a,', '    cudaEvent_t ev[10]; for(auto& e:ev) cudaEventCreate(&e); cudaEventRecord(ev[0]);\n    check(cudaMemcpy(s.a,')
profile=profile.replace('    dim3 full(', '    cudaEventRecord(ev[1]);\n    dim3 full(')
profile=profile.replace('    if(s.n<=65536)', '    cudaEventRecord(ev[2]);\n    if(s.n<=65536)')
profile=profile.replace('        middle_stages<<<', '        cudaEventRecord(ev[3]);\n        middle_stages<<<')
profile=profile.replace('        if(s.n>NTT_TILE) high_stages<true>', '        cudaEventRecord(ev[4]);\n        if(s.n>NTT_TILE) high_stages<true>')
profile=profile.replace('    if(s.crt_weights) {', '    cudaEventRecord(ev[5]);\n    if(s.crt_weights) {')
profile=profile.replace('        tensor_product<false>', '        cudaEventRecord(ev[6]);\n        tensor_product<false>')
profile=profile.replace('        if(s.gap&&s.wide<=128)', '        cudaEventRecord(ev[7]);\n        if(s.gap&&s.wide<=128)')
profile=profile.replace('    check(cudaGetLastError()', '    cudaEventRecord(ev[8]);\n    check(cudaGetLastError()')
profile=profile.replace('"copy answer"); return out;', '"copy answer"); cudaEventRecord(ev[9]); cudaEventSynchronize(ev[9]); for(int i=0;i<9;++i) {float ms=0; cudaEventElapsedTime(&ms,ev[i],ev[i+1]); std::fprintf(stderr,"%s%.6f",i?",":"profile_ms:",ms);} std::fprintf(stderr,"\\n"); for(auto e:ev) cudaEventDestroy(e); return out;')
path=root/'experiments/current_profile.cu';path.write_text(profile)
command=[str(cuda/'bin/nvcc'),'-std=c++17','-O3','-arch=sm_86','-I'+str(root/'src'),'-I'+str(root.parent/'muse-spark-open/.build-deps/cccl/nvidia/cu13/include/cccl'),'-cudart=shared','-cudadevrt=none','-L'+str(cuda/'lib'),str(root/'src/main.cpp'),str(path),'-o',str(root/'build/profile'),'-Xlinker','-rpath','-Xlinker',str(cuda/'lib')]
subprocess.run(command,check=True)
r=subprocess.run([str(root/'build/profile'),str(root/'tests/data/n65536_w1747')],capture_output=True,text=True,check=True)
rows=[[float(x) for x in line.split(':')[1].split(',')] for line in r.stderr.splitlines() if line.startswith('profile_ms:')]
report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'official':False,'stages':['h2d','encode','forward_high','middle','inverse_high','transpose','crt_mma','crt_reduce','d2h'],'ms':rows}
(root/'evidence/ntt_profile.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
