# Active CPU-copy timing diagnostic

Runs twelve complete, exact implementations with different stream/chunk/CPU-copy
schedules. Each mode consumes the current operands and finishes all work before
returning. All outputs must match, and FHERMA verifies the returned output.

The printed per-mode times and CUDA-event stage spans guide scheduling choices.
The harness time includes all twelve modes and is not a leaderboard candidate.
CUDA event instrumentation adds overhead; final candidates must be measured
without this diagnostic. Input/output data and outputs are never cached.

Source derives from `candidates/pipelined-direct-crt`; arithmetic is unchanged.
The generated harness and interface are unmodified.

Three additional modes use AVX2 streaming CPU stores, with a runtime feature
check and a fence before CUDA consumes the data. Other modes use memcpy.

Workers wake once per complete call and use atomic job publication while the
call is active. All workers leave that session before the call returns and
sleep between calls. This targets the repeated wakeup cost measured by the
previous pipeline diagnostic on Blackwell.
