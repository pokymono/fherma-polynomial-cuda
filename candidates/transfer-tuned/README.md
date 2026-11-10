# Transfer-tuned exact polynomial multiplication

Candidate for unscored validation. The larger point uses DMA input staging,
canonical 31-bit RNS arithmetic with widened Shoup differences, asynchronous
CRT tile staging, and an output graph that overlaps different arithmetic stages
across strips. The smaller point reads staged pinned inputs directly on the GPU,
sharing each input tile across RNS field groups, and retains two output chains.
The generated harness and interface are unchanged. Parameter-only state is
prepared in init; all operand work, host copies and worker joins are timed.

This is not yet a ranked submission. VALIDATION.json pairs the source and binaries
with independent exact checks. Official diagnostic results, when available, are
recorded separately and must not be confused with local GPU timing.
