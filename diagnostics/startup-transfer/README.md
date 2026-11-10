# Timed startup-transfer and phase diagnostic

Unscored investigation of the idle penalty. Compare 4 KiB, 64 KiB, 256 KiB and
1 MiB startup transfers after 20 ms idle. The transfer begins inside the inner
multiplication, overlapping CPU preparation of the real inputs. Private buffers
are allocated from public shape parameters during setup. No work runs between
calls and the unchanged harness times every transfer.

Each sample follows two verified warm calls. Three rotated rounds have warm
4 KiB brackets. CPU wall-clock timestamps split host preparation and completion;
timing-enabled external CUDA graph events split input, middle and output phases.
Event nodes perturb execution, so these measurements are diagnostic rather than
candidate scores. Input span includes queueing and any CPU starvation. The CPU
complete field ends before logging; outer_inner_ms includes logging and is not
the comparison metric. The first two outer warmups suppress logs.

Every inner answer is compared. The harness independently verifies the returned
answer. Input reuse within an outer call affects cache residency; no operand or
answer is retained across calls. Idle effects alone do not identify a power-state
mechanism. Per-call host registration and background warming are not used.

Official run `6af37583698e99b16a7cd993` passed 3/3 cases. The API retains
only the final 12,000 characters of logs, so RESULTS.json summarises only the
complete final measured outer call (three delayed samples per size, six warm
controls). Treat the apparent size differences as exploratory. Warm 4 KiB
calls had a 1.043658 ms median; delayed 4 KiB calls 1.575980 ms. Larger startup
copies did not consistently restore warm performance. The middle kernel stayed
near 64 microseconds while transfer and CPU phases changed substantially.

The input_span marker ends at strip 15, which need not finish last across
independent streams. Use input_start_to_middle, the combined adjacent intervals,
for a robust input-path comparison. No candidate was promoted from these data.
The next diagnostic uses compact logs to preserve all measured cases.
