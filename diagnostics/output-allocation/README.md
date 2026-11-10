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


The official diagnostic at commit `a408b480c306ff8499d8e54b4533fdfb182d7f5e`
passed 3/3 cases on 10 November 2026. The control measured 1.006 ms warm and
1.021 ms after 20 ms idle; huge-page advice measured 1.007 and 1.027 ms.
Allocator placement on a copy-worker sibling worsened both allocation and
full-call latency. The repeated control measured 1.005/1.070 ms, showing idle
variation. None of these policies establishes a substantial performance gain.

Full-call and allocation arrays are complete. The platform shortened long log
lines, dropping the tail of some wake-delay arrays; missing wake samples are
represented as null in RESULTS.json, never inferred.
