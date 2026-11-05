"""Use the existing local CUDA toolchain; official builds use CMake in cuPQC."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
cuda = root.parent / "muse-spark-open/.build-deps/nvcc/nvidia/cu13"
(root / "build").mkdir(exist_ok=True)
command = [str(cuda / "bin/nvcc"), "-std=c++17", "-O3", "-arch=sm_86",
           "-I"+str(root.parent / "muse-spark-open/.build-deps/cccl/nvidia/cu13/include/cccl"),
           "-cudart=shared", "-cudadevrt=none", "-L"+str(cuda / "lib"),
           str(root / "src/main.cpp"), str(root / "src/solve.cu"),
           "-o", str(root / "build/solution"), "-Xlinker", "-rpath",
           "-Xlinker", str(cuda / "lib")]
with (root / "evidence/build.log").open("w") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
print((root / "evidence/build.log").read_text())
sys.exit(result.returncode)
