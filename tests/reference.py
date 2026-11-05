"""Independent exact references and input generation for the negacyclic ring."""
import json
import random
from pathlib import Path

import gmpy2


def modulus(n, width):
    step = 2 * n
    q = ((1 << width) - 2) // step * step + 1
    while q > 1 and not gmpy2.is_prime(q, 40):
        q -= step
    if q <= 1:
        raise ValueError("No modulus exists for this point")
    return q


def schoolbook(a, b, q):
    n = len(a)
    out = [0] * n
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            k = i + j
            out[k % n] += x * y if k < n else -x * y
    return [x % q for x in out]


def kronecker(a, b, q):
    """Integer multiplication with enough zero padding to prevent carries.

    Each ordinary coefficient is at most N*(q-1)^2. Base 256**stride
    strictly exceeds that bound, so every product slot is an exact coefficient.
    This reference shares neither the GPU's RNS primes nor its NTT or CRT.
    """
    n = len(a)
    stride = ((n * (q - 1) ** 2).bit_length() + 7) // 8
    left = gmpy2.mpz(int.from_bytes(b"".join(x.to_bytes(stride, "little") for x in a), "little"))
    right = gmpy2.mpz(int.from_bytes(b"".join(x.to_bytes(stride, "little") for x in b), "little"))
    raw = int(left * right).to_bytes(2 * n * stride, "little")
    return [(int.from_bytes(raw[i * stride:(i + 1) * stride], "little")
             - int.from_bytes(raw[(i + n) * stride:(i + n + 1) * stride], "little")) % q
            for i in range(n)]


def save_point(path, n, width, seeds=(1, 2), edge_cases=True):
    path = Path(path)
    q = modulus(n, width)
    limbs = (width + 31) // 32
    byte_width = 4 * limbs
    pairs = []
    if edge_cases:
        zeros, ones, maximum = [0] * n, [1] * n, [q - 1] * n
        impulse, shifted = zeros.copy(), zeros.copy()
        impulse[0] = 1
        shifted[-1] = q - 1
        pairs.extend([(zeros, maximum), (ones, maximum), (maximum, maximum), (impulse, shifted)])
    for seed in seeds:
        rng = random.Random(seed)
        pairs.append(([rng.randrange(q) for _ in range(n)], [rng.randrange(q) for _ in range(n)]))
    (path / "point").mkdir(parents=True, exist_ok=True)
    (path / "point/q.bin").write_bytes(q.to_bytes(byte_width, "little"))
    (path / "manifest.json").write_text(json.dumps({"point": {"N": n, "W": width, "L": limbs}, "cases": len(pairs)}))
    for index, (a, b) in enumerate(pairs):
        directory = path / "cases" / f"{index:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        for name, values in (("a", a), ("b", b)):
            (directory / f"{name}.bin").write_bytes(b"".join(x.to_bytes(byte_width, "little") for x in values))
        expected = kronecker(a, b, q)
        if n <= 64:
            assert expected == schoolbook(a, b, q)
        reference = path / "reference" / f"{index:06d}"
        reference.mkdir(parents=True, exist_ok=True)
        (reference / "c.bin").write_bytes(b"".join(x.to_bytes(byte_width, "little") for x in expected))
    return {"N": n, "W": width, "L": limbs, "cases": len(pairs), "q_gap": (1 << width) - q}


def verify_point(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    reports = []
    for index in range(manifest["cases"]):
        expected = (path / "reference" / f"{index:06d}" / "c.bin").read_bytes()
        actual = (path / "out" / f"{index:06d}" / "c.bin").read_bytes()
        passed = expected == actual
        reports.append({"case": index, "passed": passed})
        if not passed:
            word = next((i for i in range(min(len(expected), len(actual))) if expected[i] != actual[i]), None)
            raise AssertionError(f"{path}: case {index} differs at byte {word}; lengths {len(actual)} / {len(expected)}")
    return reports


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("make", "verify"))
    parser.add_argument("directory")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--no-edges", action="store_true")
    args = parser.parse_args()
    result = (save_point(args.directory, args.n, args.width, range(args.seeds), not args.no_edges)
              if args.mode == "make" else verify_point(args.directory))
    print(json.dumps(result))
