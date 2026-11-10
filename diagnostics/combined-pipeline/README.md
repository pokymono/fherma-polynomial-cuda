# Combined exact pipeline diagnostic

All modes use ordered input DMA, 32-column boundary strips with64-column
interiors, and ordinary output allocation scheduled after input dispatch3.
The controls retain separate inverse, transpose, CRT and final reduction.
Other modes compare inverse/transpose fusion, the independently checked
integer tensor quotient, aligned CRT dot rows, and TMA matrix staging.
TMA descriptors and constant padding depend only on public parameters.
All operand work, allocation, transfers and completion stay inside each call.

Each outer call executes189 complete multiplications and compares every
output. There are63 measured inner calls, including warm and20ms idle
samples, rotating mode order and two controls. The returned output is also
checked by an independent harness oracle. Execution logs distinguish actual
cluster/shared fusion and TMA from fallback. These are unscored diagnostics,
not standalone timings or ranked results.

Official unscored run `6af3abe4698e99b16a7f7d17` passed all3 outer cases
on commit `6298f458bbe707aecbfde08c4ed653eb2e3577a0`. All189 measured
inner calls are in `RESULTS.json`, with actual TMA and cluster execution
confirmed. Warm medians were0.988962ms for the control,0.982090ms for
shared fusion plus quotient,0.981733ms for cluster fusion,0.980517ms
for cluster plus quotient,0.983470ms for aligned dots and0.979980ms
for TMA plus shared fusion and quotient. The repeated control was0.987170ms.
Idle medians remained substantially slower and variable. This establishes a
small combined warm-call improvement, not a standalone or ranked result.
