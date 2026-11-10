# Ordered input DMA diagnostic

This unscored experiment gives input DMA its own ordered stream and links
each copied strip to its encoder with captured external CUDA events. The
control uses the same 32-column boundary and 64-column interior input layout.
Two or eight compute streams are compared. Two additional modes postpone
the ordinary timed output allocation until input-dispatch milestones.
No GPU work occurs between inner calls, and all allocation, input packing,
transfers, arithmetic, output scattering and synchronization stay timed.

Each outer call performs 162 complete multiplications and compares every
output in full. The returned output also has an independent harness oracle.
There are 54 measured calls per outer call, split between warm and 20 ms idle
samples; the mode order rotates and includes a repeated control. These
inner diagnostics are not standalone or ranked latency measurements.

Official unscored run `6af3a860698e99b16a7f4b1b` passed all three outer cases
on commit `e78b0c70e69f646aef32e71594e5ed428591ea38`. All 162 measured
inner calls are retained in `RESULTS.json`. Warm medians were 0.998747 ms
for the first control, 0.991015 ms for eight compute streams, 0.991841 ms
for two streams, 0.989862 ms when allocation followed input dispatch seven,
and 0.988897 ms when it followed dispatch three. The repeated control was
0.997488 ms. Idle results were variable; no idle improvement is established.
This supports a modest warm-call scheduling gain, not a ranked score.
