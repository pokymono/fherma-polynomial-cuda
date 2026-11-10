# Pipelined exact modular CRT

This candidate replaces full-width integer reconstruction with an exact CRT
directly modulo the supplied coefficient modulus. At the main competition point,
the tensor reconstruction has 64 padded word columns instead of 128. It uses
integer arithmetic throughout the fast path; see [the proof](EXACTNESS.md).

Input staging, input conversion, output reconstruction, and transfers run in
chunks on nonblocking CUDA streams. A bounded CPU worker team copies the current
operands into pinned staging buffers. Output vector initialization overlaps the
GPU work; all copies and workers join before returning. CUDA graphs capture
parameter-only execution plans during initialization without executing kernels.
No operand-dependent computation, input caching, or output caching happens there.

The NTT retains the register and warp-shuffle stages of `register-ntt` and uses
subtractive Montgomery multiplication. The shortened CRT quotient is fused into
the final carry and modular reduction kernel. Other parameter sizes use exact
fallbacks.

Build this directory with the repository's normal CMake instructions. The
generated `main.cpp` and `fherma.h` remain byte-identical to `src/`.
For FHERMA, the artifact path is `candidates/pipelined-direct-crt`.

`local-results.json` records local exactness, build provenance, and timings.
Local RTX 3050 measurements do not predict an official Blackwell score.
