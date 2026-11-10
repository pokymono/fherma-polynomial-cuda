# CPU packing window and transfer diagnostic

This is a diagnostic, not a production leaderboard candidate. Each official
timed call computes the full multiplication ten times, compares every output,
and returns the first. The generated harness is unchanged.

Modes compare one, two, four, or sixteen packed input strips ahead of GPU
submission, with ordinary CPU stores and AVX2 streaming stores. One eight-worker
queue and one GPU allocation set are shared by every mode. A reference middle
NTT and a warm-execution comparison help isolate compute and startup effects.

Nine modes include a 50 ms GPU-idle interval before their inner measurement.
These idle intervals are INSIDE the official timed call. They avoid comparing
only back-to-back, warm GPU executions. The final mode omits the interval.
The official outer duration includes all idle periods, full multiplications,
comparisons, and logs; it is not competitive production latency. Inner timings
are diagnostic leads and require a separate standalone official evaluation.

Logs include stage times and available CPU topology. No operands or answers
are computed before the timed call. All CPU and GPU work joins before return.

All ten modes passed 48 fresh independent local cases (480 full computations).
SM120 compilation passed. The streaming-copy queue also passed 15,360 CPU jobs
across six team sizes, alignment boundaries, empty rows, and repeated ring reuse.
