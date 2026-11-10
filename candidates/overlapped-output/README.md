# Overlapped output construction

Extends `candidates/polling-direct-crt` with one persistent output-allocation
worker. Every call still creates and initializes a fresh standard output vector
inside the timed operation. The request joins before output access or vector
destruction and propagates allocation failures. A bounded team of seven copy
workers and one allocation worker polls only job counters between calls; no
operand or output pointer is accessed without a job from the active call.
All workers join at state destruction.

A 4 KiB transfer from a private zero-filled buffer starts inside each timed call
while CPU staging begins. It does not use a prior operand, prepare a future case,
or perform any GPU work between calls. Its completion is ordered on the compute
stream and included in the call's total time.

The exact integer CRT and NTT arithmetic is unchanged from the preceding
40/40-correct official candidate. See [the proof](../pipelined-direct-crt/EXACTNESS.md).
This exact source separately passed 138 local exact cases, 1003 output worker
construction/lifetime/error-recovery checks, and a Blackwell compilation with no
register spills. Those checks do not establish an official speed improvement.
The generated harness and interface remain unchanged.

FHERMA artifact path: `candidates/overlapped-output`.
