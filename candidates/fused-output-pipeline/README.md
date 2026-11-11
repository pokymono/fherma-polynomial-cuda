# Fused output pipeline

Exact negacyclic polynomial multiplication with 31-bit RNS fields, integer
Tensor Core conversion, fused NTT stages and centered CRT reconstruction.
The main point combines ordered input DMA, an early transform of operand A,
shared-memory inverse/transpose fusion, and the exact tensor quotient.
Its output graph runs one arithmetic strip at a time while overlapping DMA.
The smaller point retains two output arithmetic chains.

The quotient uses two extra integer tensor columns. If D0 and D1 are the
byte-column dots, D0 + 2^32 D1 equals the sum of r_k floor((2^64-1)/p_k).
Rounding (D1 + (D0 >> 32))/2^32 recovers the centered CRT quotient exactly
under the established product bound P > 4 N(q-1)^2. No floating-point
approximation or input-specific precomputation is used.

Output allocation, input staging, all GPU work, transfers and output scatter
remain within each call. Setup depends only on the public parameters.
No device work runs between calls. Hardware-dependent low-level features
retain portable fallbacks. TMA is disabled in this candidate.

This is a standalone candidate under validation. Its paired diagnostic
warm timings are not a standalone score or a ranked result.

The exact candidate passed 141 local cases and 48 independent AWS cases.
AWS medians were 1.066672 ms at N65536/W1747 and 0.348920 ms at
N32768/W868 on g7e.4xlarge with seven copy workers. These are proxy
measurements. Source hashes, commands and individual timings are in
VALIDATION.json; the official standalone check is still pending.

Official fresh-call unscored run `6af3b239698e99b16a7fa562` passed 40/40.
N65536-W1747: 1.270009 ms median.
N32768-W868: 0.398521 ms median.
The main-point result regressed despite the warm diagnostic lead. It has
not been promoted to the ranked leaderboard. See OFFICIAL.json.
