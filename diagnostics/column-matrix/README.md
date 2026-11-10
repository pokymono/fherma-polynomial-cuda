# Column pipeline diagnostic

This is a diagnostic implementation, not a standalone leaderboard candidate.
Every timed call computes the complete multiplication ten times, compares all
outputs exactly, and returns the first result. The generated harness is unchanged.

Six graph plans compare 4, 8, and 16 column strips, with and without CUDA L2
persistence for RNS intermediates. Four additional configurations compare CPU
copy lanes and CUDA streams. One shared GPU allocation set, one 16-lane copy
worker pool, and one output-allocation worker serve all modes. Workers poll only
job counters between calls. GPU work runs only inside the timed call.

The implementation logs input, middle-transform, output, CPU-staging, and total
intervals. CUDA events perturb timing slightly. Outer harness time includes all
modes and is not comparable to production latency. Mode timings are diagnostic
leads that require independent standalone evaluation.

All ten modes matched exact independent references for 48 fresh local cases
across both challenge sizes (480 full computations). Blackwell SM120 compilation
passed. The local GPU has only 1.5 MiB L2, so local cache-policy timings cannot
predict the target GPU.
