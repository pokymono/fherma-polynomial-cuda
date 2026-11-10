# Register transform and direct input conversion

This candidate combines three changes on top of the queued column-copy design:

- Input conversion reads the original coefficient limbs directly, removing the
  GPU padding kernel. The public 55-limb and 28-limb shapes have compile-time
  loop bounds; all coefficient values are still read and computed every call.
- A larger shared-memory tile reduces conversion barriers. Packed operand pairs
  use one H2D transfer per strip. GPU loads reverse the packing explicitly.
- Each warp computes a complete 512-point middle NTT in registers. Index bits
  above the warp width select registers; low bits exchange values through warp
  shuffles. It computes both forward transforms, the product, and the inverse
  transform. The outer inverse stage still applies the exact normalization.

The middle transform uses no shared-memory barriers and compiled without spills
for SM86 and SM120. Generic fallbacks remain available. Reconstruction follows
[the exact integer CRT proof](../pipelined-direct-crt/EXACTNESS.md).

All packing, allocation, transfers, transforms, reconstruction, and output
scattering remain inside `fherma_run`. Persistent CPU workers poll only job
counters between calls. There is no reuse of a previous call's answer.

This source passed all 138 independent local exact cases. Its CPU queue passed
15,360 jobs across six team sizes. Local main-case runtime fell from about
10.86 ms for the queued-copy version to about 9.31 ms in a fresh-case comparison;
that is development evidence, not a Blackwell score. A standalone official run
is required before reporting competitive latency for this candidate.

FHERMA artifact path: `candidates/register-transform`.
