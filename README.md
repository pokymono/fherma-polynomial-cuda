# Exact CUDA polynomial multiplication

Implementation of FHERMA's `polynomial-multiplication/negacyclic@1.0.0`:

```
c = a * b modulo (X^N + 1, q)
```

Inputs and output are canonical, coefficient-major arrays of little-endian
32-bit limbs. The two competition points are `(N=65536, W=1747)` and
`(N=32768, W=868)`.

## Algorithm

Parameter-only initialization chooses independent 31-bit NTT primes with product
`P > 2*N*(q-1)^2`. Each call performs the complete input transfer, coefficient
conversion, negacyclic transforms, exact CRT reconstruction, canonical reduction,
and output transfer.

Unsigned integer tensor cores evaluate byte dot products for both coefficient
conversion and the wide CRT sum. Each dot product stays within signed 32-bit
range. Carries and modular reduction are explicit integer arithmetic. A floating
point estimate is used only to propose the CRT quotient; integer comparisons and
corrections establish the exact representative before reduction modulo `q`.

The fused transforms keep groups of butterfly stages in shared memory. Generic
fallbacks handle small transforms and configurations beyond the tensor tiling
limits. No operand-dependent computation happens in initialization.

## Build and platform settings

```
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requires a C++17 compiler and a CUDA toolkit with support for the target GPU.
The tensor implementation uses unsigned INT8 WMMA (SM75 or newer). CMake defaults
to the build host's native CUDA architecture; it can be overridden with
`-DCMAKE_CUDA_ARCHITECTURES=120` for Blackwell.

For FHERMA, select the **cpp** harness, the **cupqc** image, this repository at an
exact commit, and artifact path **src**. The generated `main.cpp` and `fherma.h`
are unmodified. FHERMA replaces them with its official harness when benchmarking.

## Local verification

`tests/reference.py` uses independent GMP integer multiplication with carry-free
coefficient packing, with a schoolbook cross-check for small cases. The local
suite includes zero, unit, maximal, impulse, and random inputs across 11 points,
including both competition sizes.

```
python tools/validate_local.py
```

`tools/build_local.py` locates the existing local CUDA installation used during
development. Other machines should use CMake. `tools/profile_local.py` creates a
separate instrumented source under `experiments/`; it is never a submission file.

Local RTX 3050 timings measure development progress only. An official result
requires a passing FHERMA run on its reference runner. No official result is
claimed by this repository.
