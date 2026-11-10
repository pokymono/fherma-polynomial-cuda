# Link bandwidth diagnostic

Unscored diagnostic for the actual competition runner. The unchanged generated harness receives the exact polynomial product from the ordered-streaming implementation. Additional graph-based transfers run after that product is fully joined, using current-call staging data; no operands or answers persist across calls.

Reports CUDA event timing for 16 sequential copies, median of three measurements after one warmup. Covers input/output strip sizes, a complete polynomial, and both complete inputs; compares write-combined and cached pinned host allocations. These measurements exclude diagnostic graph construction and are not full multiplication times or leaderboard scores. Host-to-device and device-to-host figures are measured separately, not summed simultaneous bandwidth.
