# Timed transform-constant prefetch diagnostic

Compare zero,64KiB,1MiB,8MiB and up to32MiB of forward-root prefetch inside each
complete multiplication. The requested size is clamped to the allocated table.
A sixth mode prefetches the first32-column transform strip across all fields.
Only immutable, public-parameter-derived constants are accessed. The kernel
starts inside the timed call, overlaps CPU packing, and is joined before return.
No GPU activity or operand access occurs between calls; no clocks or machine
settings change. The unchanged harness also times the entire diagnostic.

Each mode measures three warm/20ms-idle/warm rounds after two verified warm calls.
Mode order rotates between rounds; one State is reused, keeping parameter setup
and worker placement identical. Every answer is compared with the first result;
the normal harness independently verifies the returned output. Input reuse is
restricted to the current outer call. Inner timings are diagnostic, not standalone
or ranked scores. All logs are emitted after measurement to preserve the idle gap.

The cache hint follows NVIDIA's PTX prefetch.global.L2 instruction contract:
https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html

## Measured result

All six modes passed 3/3 outer oracle checks and all inner comparisons. No prefetch size removed the 20ms idle slowdown: all 9 idle samples per mode exceeded 1.2ms. Warm medians increased from 1.006ms without prefetch to 1.015-1.019ms. Reject this as an idle-latency fix. See [all samples](RESULTS.json). These are unscored inner-call diagnostics, not standalone or ranked results.
