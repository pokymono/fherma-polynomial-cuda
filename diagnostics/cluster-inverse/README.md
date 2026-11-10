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

## Official diagnostic result

Run `6af39ea6698e99b16a7e8df7` passed all three independent cases and
confirmed real cluster execution for every enabled mode. Warm medians were
1.004189 ms for the control, 0.998329 ms for four fields by sixteen columns,
1.001242 ms for two fields by thirty-two columns, and 0.998668 ms for four
fields by thirty-two columns. The repeated control was 1.004952 ms.
The fusion gives a small improvement in this diagnostic, not a large gain
or a ranked result. Idle measurements remain variable. `RESULTS.json`
contains all 135 measured inner calls.
