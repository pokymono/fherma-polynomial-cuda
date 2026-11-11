# Caller construction with eight copy workers

This varies only the requested number of CPU copy workers in caller-fused-strips,
from four to eight. The worker count is capped by available physical CPU cores.
The official runner has more physical cores than the AWS development machine;
therefore an official standalone check is needed to choose the worker count.
Every input copy, output construction and GPU operation remains inside the call.
No ranked submission is associated with this candidate.

The [official standalone unscored run](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6af3d25a698e99b16a813b33) passed **40/40**.
N32768-W868 median: **0.344469 ms**.
N65536-W1747 median: **1.2838375000000002 ms**.
This is slower than the four-worker candidate; it is not selected.
