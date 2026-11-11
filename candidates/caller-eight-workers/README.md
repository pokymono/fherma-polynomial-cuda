# Caller construction with eight copy workers

This varies only the requested number of CPU copy workers in caller-fused-strips,
from four to eight. The worker count is capped by available physical CPU cores.
The official runner has more physical cores than the AWS development machine;
therefore an official standalone check is needed to choose the worker count.
Every input copy, output construction and GPU operation remains inside the call.
No ranked submission is associated with this candidate.
