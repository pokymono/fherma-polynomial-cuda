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

Official unscored run `6af3af0e698e99b16a7f8cd1` passed3/3 outer cases on2026-11-10.
Warm medians: control0.991442ms, serial0.976912ms, serial+TMA0.991580ms,
mask5 0.990616ms, TMA0.985541ms, control repeat0.992532ms. The20ms-idle
measurements were variable (1.20–1.40ms) and do not establish a stable
idle improvement. Full measurements and exact commit are in RESULTS.json.
The serialized non-TMA graph is the warm lead; it still needs a standalone
fresh-call official check. These values are not leaderboard scores.
