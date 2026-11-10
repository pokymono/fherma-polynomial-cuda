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
