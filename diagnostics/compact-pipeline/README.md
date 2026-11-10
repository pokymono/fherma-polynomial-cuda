# Compact tables and early-operand pipeline comparison

Unscored comparison of ordered-streaming, transfer-tuned, compact high-stage root
tables, and the compact pipeline with an early forward transform of operand A.
Baseline measurements bracket the other modes. The original outer harness is
unchanged. Each mode performs three warm/20ms-idle/warm rounds, each preceded by
two verified warm calls. Inner timings exclude initialization; the outer measured
call includes the entire diagnostic. These are not standalone or ranked scores.

All worker threads use the existing UMONITOR/UMWAIT policy. No operand work or GPU
activity occurs between inner calls. The inputs are reused only within the current
outer call. Every inner answer is compared with the baseline, and the returned
answer is checked independently by the normal harness.

The compact table stores q=floor(root*2^32/p) alone and recovers the nonzero root
exactly as high32(q*p)+1. For odd p and 0<root<p, the remainder in root*2^32-q*p
is strictly between0 andp, proving the identity. Parameter-only high-stage root
tables shrink by half. This identity passed1261632 independent integer checks.

The early-operand variant transfers eight64-column strips of A followed by B.
A's central forward transform overlaps B's transfers. Output retains sixteen
32-column strips. Compact roots, asynchronous CRT loads, and output stage
scheduling are combined with the previously validated arithmetic. The smaller
point retains the existing mapped-input path.

Before this diagnostic, both new variants passed48 exact local cases and48
independent Blackwell cases each. The comparison wrapper also passed one local
oracle case. AWS complete-call measurements on a4-physical-core g7e.2xlarge
were1.056030ms for compact-tuned (matched original1.067076ms), and1.0536795ms
for compact-operand (compact controls1.0613095/1.0570755ms). These modest gains
have not been promoted to the ranked implementation.

## Official unscored result

Run6af3813b698e99b16a7d1590 finished2026-11-10T20:41:30.267Z with3/3
correct cases. All135 measured inner samples survived the log limit. Warm
medians were1.034186/1.035471ms for the bracketing baseline,1.022276ms for
transfer-tuned,1.0165465ms for compact roots, and1.0043255ms for compact plus
early operand transform. This confirms modest warm-call gains on the official
GPU. The20ms-idle medians remained1.25–1.38ms, with substantial variation;
neither arithmetic change resolves that slowdown. See RESULTS.json for every
sample, counts and ranges. No ranked or standalone claim is made.
