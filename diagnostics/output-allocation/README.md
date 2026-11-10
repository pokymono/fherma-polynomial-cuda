# Output allocation diagnostic

Unscored comparison of default allocation, large-page advice restricted to
the interior of this call's owned output allocation, normal-page advice,
and placement of the allocator on a sibling of the caller or first copy worker.
The source uses compact roots and early first-operand transformation.

All output allocation, initialization, input copies, arithmetic and output
copies remain inside each measured inner call. No machine settings or global
allocator policy are changed. Every inner output is compared in full, and the
outer result goes to the independent official verifier. Mode order rotates.
Each state is destroyed before changing its allocation policy.

Each outer call contains 189 complete inner calls: seven modes, three rounds,
three positions, and two verified warmups plus one measurement per position.
Positions are warm, 20 ms idle, and warm. Alongside complete-call latency, the
diagnostic reports allocation work and allocator wake delay. These measurements
are not standalone leaderboard scores.
