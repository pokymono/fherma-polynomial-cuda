# Transform-aligned column streaming

This candidate streams complete columns of the first NTT decomposition instead
of consecutive coefficient ranges. A CPU worker team packs each column strip
inside the timed call. The GPU converts that strip and immediately performs its
independent forward transform, overlapping this work with later input transfers.
Only the middle transform waits for every input strip.

Each output stream then performs the inverse transform for its columns,
transposes those residues into a packed CRT layout, reconstructs exact canonical
coefficients, and returns that strip. The CPU scatters it into the required
coefficient-major output vector while later GPU strips are still processing.

The mathematical transform is unchanged. The packing permutation is:

```
packed = chunk * (N/16) + row * 32 + column
logical_coefficient = row * 512 + chunk * 32 + column
```

Input conversion applies twists using the logical index. Forward and inverse
high stages use the same logical column offset. The gathered transpose and
output scatter reverse this permutation. All 16 strips partition the full
coefficient range; no result depends on a previous call.

CRT exactness follows [the direct-CRT proof](../pipelined-direct-crt/EXACTNESS.md).
The generated harness and interface are unchanged. Allocation, packing,
transfers, transforms, CRT, and scattering all finish within `fherma_run`.
Eight persistent CPU workers poll only job counters between calls and join at
state destruction. The private startup transfer is also inside the timed call.

This exact source passed 138 local exact cases. The strided copy helper passed
96 additional jobs over 48 sessions, including empty and uneven partitions.
Local timing is not an official Blackwell score. The 16-strip version was kept
over an 8-strip experiment because it was faster on the development GPU.

FHERMA artifact path: `candidates/column-streaming`.
