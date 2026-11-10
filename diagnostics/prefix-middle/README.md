# Streamed prefix-transform diagnostic

Unscored comparison of the compact-root/early-operand pipeline with an exact
64-by-8 factorization of its middle transform. Six forward stages run for each
input strip while later strips transfer. The final three forward stages,
pointwise products, and full inverse execute after the input join.

Modes compare the control, ordinary interleaved packing, AVX-512 packet packing
with eight, ten or six requested copy workers, and a repeated control. All
policies are fixed per state; a state is destroyed before the next policy starts.
The CPU feature check retains ordinary packing where AVX-512 is unavailable.

Each outer call runs 162 complete inner multiplications: six modes, three
rounds, three positions and two verified warmups plus one measurement. The
positions are warm, 20 ms idle, and warm. Every result is compared in full;
the returned result is independently verified by the official harness.
All operand-dependent work stays timed. No background GPU work is performed.
These are diagnostic inner-call measurements, not standalone ranked scores.

## Official diagnostic result

Run `6af399d8698e99b16a7df684` passed all three independent cases. The
162 measured inner calls gave warm medians of 1.003810 ms for the control,
1.018943 ms for ordinary prefix packing, 1.024035 ms for vector packing,
and 1.015460 ms with ten copy workers. The repeated control was 1.004619 ms.
The prefix arrangement therefore did not improve this official diagnostic.
Idle measurements varied substantially; these are not leaderboard scores.

On the separate eight-core AWS host, plain prefix packing improved a seven-
worker run from 1.092012 ms to 1.048307 ms. Paired Nsight traces attribute most
of that difference to faster H2D transfer, despite slower CPU packing; the
final middle kernel saved about 5 microseconds. This machine-specific result
does not override the official comparison. See `RESULTS.json` for every
measured official sample and the source commit.
