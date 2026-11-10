# Clustered inverse-transform diagnostic

Unscored comparison of the existing exact pipeline with an inverse high NTT
that writes the CRT layout directly through distributed shared memory.
Two or four thread blocks exchange residues and issue coalesced field stores.
This removes the separate transpose and its intermediate global-memory pass.
All cluster participants synchronize before remote reads and before exit.

Five configurations include two controls and three cluster shapes. Each outer
call runs 135 complete multiplications, including 45 measured inner calls.
Every output is compared in full and the returned output is independently
verified by the official harness. Fixed policies are selected before setup;
all operand-dependent work remains timed. Cluster execution is logged, and
unsupported hardware uses the ordinary pipeline. There is no background GPU
work. Inner timings are not standalone leaderboard measurements.
