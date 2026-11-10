# Host memory and worker diagnostic

This diagnostic measures twelve complete, exact polynomial multiplications
inside each `fherma_run`, comparing every result. Its outer harness time includes
all modes, worker reconstruction, output comparisons, and twelve 50 ms idle
periods. The inner measurements are diagnostics, not leaderboard scores.

The CUDA path is the refined column implementation. The matrix compares actual
teams of 2, 4, 8, or 12 copy lanes (including the caller); ordinary, cached AVX512,
or streaming AVX512 input stores; cacheable or write-combining pinned input
buffers; and guarded WAITPKG waiting. Output copies always use ordinary memory
copies. A store fence completes each lane's write-combining or streaming writes
before CUDA can consume the strip. Unused CPU workers are destroyed, rather
than left polling during a smaller-team measurement.

Both input-buffer allocations and captured plans depend only on public
parameters. Operands are copied only inside the timed call. There is no GPU
work between calls or during the diagnostic idle periods. Neither the harness,
CPU affinity, system power settings, nor allocator behavior is modified.

The local check passed 48 independent cases at both challenge sizes, with all
twelve mode outputs equal in every case (576 full computations). The local
Ryzen 6800H has neither AVX512F nor WAITPKG, so those guarded branches use their
fallbacks locally and require exact checks on a capable runner. SM120 compilation
also passed. See [local results](local-results.json).

The memory-type experiment follows NVIDIA's
[write-combining documentation](https://docs.nvidia.com/cuda/archive/11.1.1/cuda-c-programming-guide/index.html#write-combining-memory).
The monitored wait uses C0.1 with a bounded deadline and a recheck after arming
the address monitor, following Intel's
[instruction specification](https://cdrdv2-public.intel.com/789573/252046-sdm-change-document.pdf).
Hardware support is checked at runtime. These references motivate experiments;
they do not establish a speedup for this workload.

Artifact path: `diagnostics/host-memory-matrix`.
