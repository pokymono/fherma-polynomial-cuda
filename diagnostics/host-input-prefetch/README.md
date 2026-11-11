# Fresh-input CPU prefetch diagnostic

Each supplied input is multiplied once. Four host input-copy implementations
rotate across calls: existing AVX-512 streaming stores (0), 512-byte T0 read-ahead
(1), 2048-byte T0 read-ahead (2), and 2048-byte NTA read-ahead (3). Each read-ahead
address stays within the current valid source row. Output allocation is immediate
in every mode. The CUDA graph and exact arithmetic remain unchanged.

The existing per-strip SFENCE orders all stores before the ready publication.
Every CPU worker and GPU operation finishes before return. No input is touched
outside its measured call, and no work persists between calls. One compact log records the mode and pre-log total time; official outer timings
include logging. There are no CPU phase timestamps or extra GPU events.
This rotated diagnostic is not a standalone or ranked candidate score.

Official run `6af3df40698e99b16a822787` passed20/20, exercising the AVX-512 paths. No prefetch mode was selected; see RESULTS.json.
