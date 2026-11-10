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
