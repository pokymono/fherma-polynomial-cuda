# Caller construction with fused output strips

Four copy workers prepare the inputs while a dispatch worker queues GPU work
and the caller constructs its ordinary output vector. The main point uses
32-column boundary strips and 64-column interior strips. The small point uses
64-column strips and a fused inverse transform and transpose covering two fields
and eight columns per block. All workers and GPU operations finish before return.
The generated harness and output ownership are unchanged.

The final source passed 141 local exact cases and 48 AWS exact cases.
`VALIDATION.json` pairs source and binary hashes with those results.

[Official standalone unscored run](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6af3cd14698e99b16a80f3ee)
passed **40/40**: main median **1.0024405 ms**, small median **0.3176965 ms**.
The main range was 0.990317–1.041868 ms. `OFFICIAL.json` records the result.
This candidate has no ranked submission.
