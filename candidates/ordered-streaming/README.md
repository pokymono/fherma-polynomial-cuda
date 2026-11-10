# Ordered streaming with exact Shoup transforms

This candidate overlaps CPU packing, input DMA, NTT stages, output DMA, and CPU
scattering. The multiplication and CRT reconstruction remain exact. It combines
the measured improvements from experiments v094, v096, v097, v099, v107, and
v108. This directory has its own validation record; another candidate's official
score does not apply to it.

Official run at source commit `a1477c968fdcf5934ca3a2ee49914e20198cf870` passed **40/40** cases,
with medians **1.0712435 ms** (main) and **0.3752785 ms** (small).
See [official-results.json](official-results.json) and the [platform run](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6af33259698e99b16a7b147a).

## Changes

Persistent input workers own disjoint rows for the entire call. A release flag
publishes each completed strip after every worker has fenced its non-temporal
stores. On x86, AVX-512 stores and monitored idle waits have runtime feature
checks. Other CPUs use the existing ordinary-copy and waiting paths.

Each strip packs A and B together for one input transfer. Eight independent
input streams overlap DMA with conversion and forward NTT stages. The output
graph has two arithmetic chains and one ordered DMA chain, allowing arithmetic
to proceed while a previous strip is transferred. The caller joins the workers
in scattering completed strips into the returned vector.

The high and middle NTT stages use canonical Shoup multiplication with
parameter-only precomputed quotients. High-stage twiddles are stored by stage
and column so adjacent threads read adjacent entries. The pointwise product and
boundary conversions retain Montgomery arithmetic. The 30-bit lazy-reduction
experiment is **not** part of this candidate: its faster kernel did not produce
a repeatable improvement in the complete call.

Workers use distinct allowed physical CPU cores where available, reserving the next
physical core for the caller. The allocation worker uses the following spare
physical core, otherwise an
allowed sibling. The caller's original affinity is restored before return.
Output allocation and initialization occur inside the measured call. All CPU
work and GPU work finish before the function returns.

## Exactness

For `0 <= a,b < p < 2^31`, let `q_b = floor(b*2^32/p)` and
`q = floor(a*q_b/2^32)`. The integer remainder `a*b-q*p` lies in `[0,2p)`;
therefore its low 32 bits are its complete value. One subtraction of `p`
produces the canonical product modulo `p`. The new tables contain the same
roots in normal representation and their exact integer quotients. Stage-major
layout changes only addresses.

The [direct CRT proof](../pipelined-direct-crt/EXACTNESS.md) and
[tensor conversion bounds](../gpu-refined-columns/TENSOR_CONVERSION.md) still
apply. Initialization depends only on public parameters. There is no
operand-dependent precomputation, cached answer, output pool, or harness change.
The generic transform path remains available outside the column-pipeline sizes.

## Reproduction

```sh
cmake -S candidates/ordered-streaming -B build/ordered-streaming-cmake \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/ordered-streaming-cmake -j2
build/ordered-streaming-cmake/solution PATH_TO_POINT
```

The official target uses CUDA 12.8 and SM120. `main.cpp` and `fherma.h` retain
the generated harness unchanged. Default tuning is eight input streams, two
output arithmetic chains, merged input copies, and at most eight input workers
adapted to the allowed physical-core count.

`VALIDATION.json` records source hashes, independent exactness checks, and AWS
measurements. AWS has the matching GPU but a different CPU and fewer CPU cores
than the official runner. Those measurements are development evidence, not a
leaderboard score.
