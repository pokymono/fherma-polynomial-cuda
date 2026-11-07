Diagnostic artifact for verified stage timing on FHERMA hardware.

The generated harness is unchanged. CUDA event recording and stderr reporting
are inside the implementation and included in measured execution. This is an
instrumented diagnostic build, not the production leaderboard candidate.
Stages: input copies, encoding, NTT, transpose, CRT MMA, CRT reduction, output copy.
