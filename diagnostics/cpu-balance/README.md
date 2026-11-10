# Copy-team balance and worker wait comparison

Unscored comparison using identical compact-root/early-operand GPU arithmetic.
Compare requested4,6,8 copy workers with the existing UMONITOR/UMWAIT policy;
also compare4 and8 workers using PAUSE polling, plus a repeated8-worker control.
Requests are capped to available physical cores minus the caller core. Actual
worker counts are logged. Caller and allocation-worker placement follow the
selected count. Thus this measures the complete scheduling policy, not just
thread count in isolation.

Each mode owns one State and joins/destroys every worker before changing policy.
Idle workers only check the job generation counter. They never access operands,
produce answers or launch GPU work between calls. Polling consumes extra CPU.
No machine settings or clocks change. All operand work remains in the timed call.

Each mode measures three warm/20ms-idle/warm rounds after two verified warm calls.
Mode order rotates across outer calls. Every answer is compared; the returned
answer is checked by the unchanged official harness. Inner times use repeated
current inputs and exclude setup, so they are not standalone or ranked scores.

## Measured result

All modes passed 3/3 outer oracle checks and all inner comparisons. Eight workers gave warm medians 1.010-1.013ms; four workers regressed to 1.208ms and six to 1.024ms. PAUSE polling did not improve the eight-worker warm median or the idle median. The AWS preference for four workers does not transfer to the official host; retain eight on the target. See [all samples](RESULTS.json). These are unscored inner-call diagnostics, not standalone or ranked results.
