# Exact tensor quotient and inverse fusion diagnostic

The CRT matrix executes a 64-column tile but uses only 57 logical columns at
the main point. Two spare columns now calculate an exact fixed-point quotient
estimate. Their byte-convolution outputs satisfy D0 + 2^32 D1 =
sum(r_k floor((2^64-1)/p_k)). Rounding after division by 2^64 gives the same
centered-CRT quotient: its true fractional part is within one quarter of the
correct integer, and reciprocal error is below sum(p_k-1)/2^64. This avoids a
separate residue read and per-warp quotient reduction. No floating arithmetic
is used. `ARITHMETIC.json` records 2016 independent identity checks, including
1010 centered-CRT margin cases with fresh integers and edge cases.

The diagnostic also compares the confirmed inverse/transpose cluster fusion,
a variant that exchanges fields inside a single block, and combined pipelines.
Optional padding elimination initializes the public, always-zero CRT padding
once, then processes only groups that contain real fields. Every real field
is overwritten from the current operands on every invocation.

Seven modes include two controls and isolated versus combined changes. Each
outer call performs 189 full multiplications, with 63 measured inner calls.
Every result is compared in full and the returned result has an independent
harness oracle. All operand-dependent work stays timed. Cluster execution is
logged; unsupported hardware uses the ordinary inverse pipeline. These are
unscored inner-call diagnostics, not leaderboard measurements.

Official unscored run `6af3a482698e99b16a7efb83` passed all three outer cases
on commit `7e3a12a045d38b15d25271f664f5ccfae6b4ce50`. All 189 measured
inner calls are in `RESULTS.json`; requested cluster/shared execution was confirmed.
Warm medians were 1.004115 ms for the control, 1.002007 ms for quotient only,
0.998662 ms for cluster only, 1.000908 ms for cluster plus quotient and padding
elimination, 0.994892 ms for the shared-block combination, and 0.997724 ms for
the full-padding cluster combination. The repeated control was recorded separately.
Idle samples varied substantially and did not establish an improvement. These
results support only a small warm-call gain, not a ranked latency claim.
