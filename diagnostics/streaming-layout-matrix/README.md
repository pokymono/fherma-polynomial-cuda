# Streaming layout diagnostic

Unscored comparison of the validated 32-column DMA baseline, 64/128-column
DMA layouts, and 32-column staging copied by 64 GPU blocks. Each mode uses
one implementation state at a time and performs five complete multiplications.
The baseline repeats at both ends to expose drift. Every answer is compared.

The unchanged platform harness times the entire call, including mode setup,
all inner runs, comparisons, destruction and logs. Inner timings exclude only
parameter-based setup and are diagnostic measurements, never leaderboard scores.
This is for the private profile implementation, not the challenge endpoint.
