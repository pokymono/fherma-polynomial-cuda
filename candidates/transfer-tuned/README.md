# Transfer-tuned exact polynomial multiplication

This experimental candidate passed140local independent exact checks,48AWS
checks and6/6official unscored checks. The official standalone median was
0.349913ms at N32768 W868 and1.155817ms at N65536 W1747. The small-case gain
reproduced; the main-case proxy/warmed diagnostic gain did not. This candidate
has not been promoted to the ranked implementation.

The larger point uses DMA input staging, widened Shoup differences,
asynchronous CRT tile staging and an output graph that overlaps different
arithmetic stages across strips. The smaller point reads staged pinned inputs
directly on the GPU, sharing each tile across RNS field groups, and retains two
output chains. The generated harness and interface are unchanged. All operand
work, host copies and worker joins remain inside the timed operation.

VALIDATION.json pairs source and binaries with independent checks; RESULTS.json
records the standalone official run and its three per-point measurements. Those
results are distinct from the repeated inner diagnostics and AWS timings.
