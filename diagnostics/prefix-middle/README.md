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
