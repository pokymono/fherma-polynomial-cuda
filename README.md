# Exact CUDA polynomial multiplication

Implementation of FHERMA's `polynomial-multiplication/negacyclic@1.0.0`:

```
c = a * b modulo (X^N + 1, q)
```

Inputs and output are canonical, coefficient-major arrays of little-endian
32-bit limbs. The two competition points are `(N=65536, W=1747)` and
`(N=32768, W=868)`.

The fastest validated official entry at the main competition point is
[`candidates/ordered-streaming`](candidates/ordered-streaming/README.md), at commit
`a1477c968fdcf5934ca3a2ee49914e20198cf870`. It passed all 40 official cases on
Blackwell with medians of **1.0712435 ms** and **0.3752785 ms**, respectively.
[Official run](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6af33259698e99b16a7b147a).
The main-point latency is 16.8% lower than the previous best 1.2879635 ms entry.
Use that directory in the CMake commands below to reproduce this version;
`src/` preserves the original baseline. Experimental measurements have separate
records and do not inherit the official result.

## Algorithm

Parameter-only initialization chooses independent 31-bit NTT primes with product
`P > 4*N*(q-1)^2`. Each call performs all host packing, transfers, coefficient
conversion, negacyclic transforms, exact CRT reconstruction, canonical reduction,
and output scattering.

Unsigned integer tensor cores evaluate byte dot products for coefficient
conversion and the wide CRT sum. Each dot product stays within signed 32-bit
range. The current fast path uses a proven fixed-point integer CRT quotient
and reconstruction modulo `q`, halving the main reconstruction width. Carries
and modular reduction use exact integer arithmetic. See the
[exactness proof](candidates/pipelined-direct-crt/EXACTNESS.md).

Transform-aligned column strips allow forward stages to overlap input transfers
and inverse stages to overlap output transfers. Generic fallbacks handle small
transforms and configurations beyond the optimized tiling limits. No
operand-dependent computation happens in initialization.

## Build and platform settings

```
cmake -S candidates/ordered-streaming -B build/reproduce -DCMAKE_BUILD_TYPE=Release
cmake --build build/reproduce -j
```

Requires a C++17 compiler and a CUDA toolkit with support for the target GPU.
The tensor implementation uses unsigned INT8 MMA (SM80 or newer). CMake defaults
to the build host's native CUDA architecture; it can be overridden with
`-DCMAKE_CUDA_ARCHITECTURES=120` for Blackwell.

For FHERMA, select the **cpp** harness, the **cupqc** image, this repository at an
exact commit, and artifact path **candidates/ordered-streaming**. The generated `main.cpp` and `fherma.h`
are unmodified. FHERMA replaces them with its official harness when benchmarking.

## Local verification

`tests/reference.py` uses independent GMP integer multiplication with carry-free
coefficient packing, with a schoolbook cross-check for small cases. The local
suite includes zero, unit, maximal, impulse, and random inputs across 11 points,
including both competition sizes. The extended `tools/stress_local.py` suite
adds 20 fresh random cases at each competition size, wraparound and carry
boundaries, wider coefficients, and a larger transform fallback.

```
python tools/validate_local.py
```

`tools/build_local.py` locates the existing local CUDA installation used during
development. Other machines should use CMake. `tools/profile_local.py` creates a
separate instrumented source under `experiments/`; it is never a submission file.

Local RTX 3050 timings measure development progress only. The production source
at commit `3f8ebaf697bf7220fed8f119e7d0184f83779f52` passed all 40 official cases
on the NVIDIA RTX PRO 6000 Blackwell reference runner on November 7, 2026.
Its median times were 5.639522 ms at `(65536,1747)` and 1.4168585 ms at
`(32768,868)`; see [the official run](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6aeee932ed7d31391ff332ef).
Experimental candidates and diagnostics have separate artifact directories and
must be validated and measured independently before those results apply to them.

The [pipelined direct-CRT candidate](candidates/pipelined-direct-crt/README.md)
uses a proven integer quotient to halve the main reconstruction width and
overlaps host staging, GPU conversion, and output transfers. It passed 138 local
exact cases and 4044 independent algebraic checks. Sequential alternating local
measurements showed 16.06 ms to 10.64 ms at the main point (1.51x); this is not an
official Blackwell result. Its artifact directory includes the mathematical
proof, source hashes, and detailed local results.
