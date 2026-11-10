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
