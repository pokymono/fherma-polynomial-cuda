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
