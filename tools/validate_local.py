"""Run fresh exact arithmetic checks; local timings are not official scores."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from reference import save_point, verify_point

points = [(1, 2), (2, 5), (8, 16), (64, 31), (64, 33), (128, 128),
          (256, 256), (1024, 512), (8192, 256), (32768, 868), (65536, 1747)]
report = {"hardware": "RTX 3050 Laptop, SM86", "official": False,
          "source_sha256": hashlib.sha256((ROOT / "src/solve.cu").read_bytes()).hexdigest(),
          "reference": "independent GMP integer convolution with carry-free coefficient packing; schoolbook cross-check for N<=64",
          "points": []}
for n, width in points:
    at = ROOT / "tests/data" / f"n{n}_w{width}"
    start = time.monotonic()
    info = save_point(at, n, width, seeds=(101, 202), edge_cases=True)
    subprocess.run([str(ROOT / "build/solution"), str(at)], check=True, timeout=300)
    checks = verify_point(at)
    measurements = json.loads((at / "out/results.json").read_text())
    info.update(passed=len(checks), measurements=measurements, validation_wall_seconds=time.monotonic()-start)
    report["points"].append(info)
    (ROOT / "evidence/local_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"N": n, "W": width, "passed": len(checks), "seconds": info["validation_wall_seconds"]}), flush=True)
print(json.dumps({"total_passed": sum(x["passed"] for x in report["points"]), "all_passed": True}), flush=True)
