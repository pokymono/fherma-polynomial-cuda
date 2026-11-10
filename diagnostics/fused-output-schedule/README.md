# Fused output scheduling diagnostic

This compares output dependency graphs after the exact inverse/transpose
fusion and tensor quotient. All modes retain ordered input DMA and timed
output allocation. The serialized-strip mode starts the next strip after
the preceding strip finishes arithmetic; output DMA can still overlap it.
Other modes serialize selected arithmetic stages across strips. TMA on/off
is compared explicitly, with execution logged.

Every outer call executes162 complete multiplications, compares every
output, and measures54 inner calls with rotating modes and two controls.
The returned output also has an independent harness oracle. All allocation,
operand work and transfers remain timed; no work runs between calls. These
unscored inner timings are not standalone or ranked latency measurements.
