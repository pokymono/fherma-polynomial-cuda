# Exact modular CRT with persistent CPU copy workers

Uses the exact integer arithmetic described in
[the direct-CRT proof](../pipelined-direct-crt/EXACTNESS.md).
The complete call includes host input staging, transfers, conversion, NTT,
CRT, canonical reduction, output vector construction, and the output copy.
The generated harness and interface remain unchanged.

A bounded team of eight CPU participants copies host buffers. The seven worker
threads poll only an atomic job counter between calls. They never inspect input
or output pointers unless the active call publishes a copy job. Each copy waits
for every participant before returning; all threads join at state destruction.
This avoids the per-chunk wakeup latency measured by the Blackwell diagnostics.
It consumes CPU resources while the implementation state exists.

The main point uses eight chunks on two nonblocking CUDA streams; the smaller
point uses four chunks. Parameter-only setup allocates buffers and captures CUDA
graphs without executing operand-dependent work. No input or output is cached.

The preceding diagnostic with sleeping workers reached approximately 2 ms per
mode on Blackwell and passed three cases. That is a different artifact and is
not a standalone score for this polling version. See local-results.json for
this exact source's independent local validation and the official platform for
its eventual standalone result.

FHERMA artifact path: `candidates/polling-direct-crt`.
