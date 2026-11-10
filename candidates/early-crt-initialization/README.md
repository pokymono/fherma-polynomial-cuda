# Early CRT initialization with the register transform

This candidate uses eight CPU copy workers, direct input conversion, paired
operand transfers, and the register-only 512-point middle transform. The
length-two butterfly, pointwise product, and first inverse butterfly are fused;
the two identity twiddle multiplications are unnecessary.

The tensor-CRT padding is initialized at the start of every timed call on the
compute stream. This real initialization overlaps CPU input packing. Later
transpose kernels write only valid residue columns. Stream dependencies ensure
padding initialization completes before any CRT kernel reads the buffer.

All computation, packing, allocation, transfers, and output scattering remain
inside `fherma_run`. No GPU work occurs between calls. Constant setup, buffers,
and execution plans remain parameter-only. The generated harness is unchanged.

This exact source passed 138 independent local cases and compiled for SM120
without spills. The CPU queue previously passed 15,360 jobs across six team
sizes. Local main-point runtime was about 9.58 ms, compared with about 9.28 ms
for the version that initializes padding later. Moving initialization earlier
is a target-hardware scheduling experiment, not a claimed local improvement.
A standalone Blackwell evaluation is required.

FHERMA artifact path: `candidates/early-crt-initialization`.
