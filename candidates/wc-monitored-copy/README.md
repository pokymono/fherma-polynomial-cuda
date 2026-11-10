# Write-combining input staging with monitored copy workers

This candidate combines the exact GPU kernels from `gpu-refined-columns` with
write-combining pinned input buffers, non-temporal AVX512 stores when supported,
and guarded WAITPKG waits for the seven copy helpers. The calling thread is
the eighth copy lane. Output allocation and output copies retain their original
behavior. CPU affinity is unchanged.

Each lane issues a store fence before publishing completion, ensuring the GPU
sees every input byte. The monitored wait arms the job counter, rechecks it, and
uses C0.1 with a bounded deadline. Unsupported CPUs use ordinary copies and
pause loops. No worker reads operands between calls. Every job finishes before
the call returns, and destruction joins the helpers.

All 138 independent local cases passed. SM120 compilation passed. The local CPU
uses the AVX512/WAITPKG fallbacks; the hardware-specific instructions were checked
by the exact [official host-memory diagnostic](../../diagnostics/host-memory-matrix/official-results.json).
That diagnostic found improvements against its own ordinary-copy control, but
does not establish this combined candidate's full official performance.

The main-point local median was about 8.84 ms on RTX 3050. The full official run passed 40/40 with medians of **1.2879635 ms** at
N65536/W1747 and **0.444297 ms** at N32768/W868. See [official results](official-results.json).

FHERMA artifact path: `candidates/wc-monitored-copy`.
