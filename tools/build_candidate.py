"""Build a candidate with the existing CUDA toolkit and save its build provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('--name')
parser.add_argument('--arch', default='sm_86')
parser.add_argument('--define', action='append', default=[])
parser.add_argument('--compile-only', action='store_true')
args = parser.parse_args()
source = args.source.resolve()
name = args.name or source.name
if Path(name).name != name:
    parser.error('--name must be a filename, not a path')
cuda = root.parent / 'muse-spark-open/.build-deps/nvcc/nvidia/cu13'
output = root / 'build' / (name + ('.o' if args.compile_only else ''))
command = [str(cuda / 'bin/nvcc'), '-std=c++17', '-O3', '-arch='+args.arch,
           '-Xcompiler=-pthread', '-Xptxas=-v',
           '-I'+str(root.parent / 'muse-spark-open/.build-deps/cccl/nvidia/cu13/include/cccl')]
command.extend('-D'+value for value in args.define)
if args.compile_only:
    command.extend(['-c', str(source / 'solve.cu')])
else:
    command.extend(['-cudart=shared', '-cudadevrt=none', '-L'+str(cuda / 'lib'),
                    str(source / 'main.cpp'), str(source / 'solve.cu'),
                    '-Xlinker', '-rpath', '-Xlinker', str(cuda / 'lib')])
command.extend(['-o', str(output)])
files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
         if p.suffix in {'.cu', '.cpp', '.h', '.hpp', '.cuh'} or p.name == 'CMakeLists.txt'}
log = root / 'evidence' / (name+'_build.log')
with log.open('w') as handle:
    result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
report = {'command': command, 'source_files_sha256': files, 'exit_code': result.returncode,
          'binary_sha256': hashlib.sha256(output.read_bytes()).hexdigest() if result.returncode == 0 else None}
(root / 'evidence' / (name+'_build.json')).write_text(json.dumps(report, indent=2)+'\n')
if result.returncode:
    print(log.read_text())
    raise SystemExit(result.returncode)
print(f'Built {output} ({args.arch}); provenance: {name}_build.json')
