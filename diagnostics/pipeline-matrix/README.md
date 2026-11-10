# Pipeline timing diagnostic

Runs nine complete, exact implementations with different stream/chunk/CPU-copy
schedules. Each mode consumes the current operands and finishes all work before
returning. All outputs must match, and FHERMA verifies the returned output.

The printed per-mode times and CUDA-event stage spans guide scheduling choices.
The harness time includes all nine modes and is not a leaderboard candidate.
CUDA event instrumentation adds overhead; final candidates must be measured
without this diagnostic. Input/output data and outputs are never cached.

Source derives from `candidates/pipelined-direct-crt`; arithmetic is unchanged.
The generated harness and interface are unmodified.
