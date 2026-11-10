# Queued column copies

This candidate keeps the exact column-streaming GPU algorithm and removes a
CPU scheduling barrier. A bounded queue lets copy workers pack later input
strips while the calling thread launches already completed strips on the GPU.
On output, workers scatter finished strips while the caller waits for later
GPU transfers. Every copy completes before `fherma_run` returns.

Each worker owns disjoint rows. Job publication and per-strip completion use
release/acquire atomics; a ring slot is reused only after every worker has
finished it. The queue never reads operands between calls. Threads join at
state destruction. Output allocation remains inside the timed call.

Machines exposing at least 20 hardware threads use 16 copy workers; smaller
machines use eight. The output-allocation worker is separate. This choice is
based only on available CPU hardware, not on case contents or prior results.

The candidate passed 138 independent local exact cases. Its queue passed
15,360 additional jobs over 480 sessions, covering six worker-team sizes,
empty rows, uneven partitions, unaligned addresses, and repeated ring reuse.
Blackwell SM120 compilation passed with zero spills. Address/undefined
sanitizer linking was unavailable because the local system ASan library is
missing; this is not reported as a sanitizer pass.

The standalone official run passed 40/40 cases. Its main-point median was
**1.4335295 ms**, slower than the prior 1.3373995 ms column implementation.
The secondary point improved to **0.4361335 ms**. This version is not promoted
as the main-point incumbent. See [the exact run record](official-results.json).

FHERMA artifact path: `candidates/queued-column-copy`.
