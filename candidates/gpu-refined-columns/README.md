# Refined GPU kernels with the original host pipeline

This candidate keeps the host scheduling from the 1.3373995 ms official
column-streaming implementation: seven copy helpers plus the calling thread,
one output-allocation worker, sixteen strips, four CUDA streams, and ordinary
CPU copies. The host helper headers are byte-identical to that implementation.

GPU changes remove input padding, use matrix-load instructions for tensor
fragments, simplify exact Montgomery conversion, and keep the 512-point middle
transform's operands in registers. Compact shared twiddle tables reduce the
middle kernel to 56 registers per thread on SM120 without spilling. The output
transpose still initializes CRT padding in its original position.

Every operand-dependent operation, allocation, transfer, transform, and output
copy remains inside `fherma_run`. There is no GPU work between calls, CPU affinity
change, operand-dependent setup, or reuse of earlier results.

The exact tensor conversion is explained in [TENSOR_CONVERSION.md](TENSOR_CONVERSION.md).
CRT reconstruction follows [the existing proof](../pipelined-direct-crt/EXACTNESS.md).
This source passed 138 independent local cases and 11,021 independent integer
conversion checks. SM120 compilation passed without spills. Local main-case
runtime was about 8.76 ms; a separate official run is required for a leaderboard
performance claim.

FHERMA artifact path: `candidates/gpu-refined-columns`.
