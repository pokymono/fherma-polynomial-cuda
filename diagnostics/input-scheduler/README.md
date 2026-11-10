# Input staging and submission diagnostic

Seven unscored modes compare AVX-512, AVX2 and MOVDIR64B staging stores;
GPU readiness gates; one complete graph; that graph with caller-assisted input
packing; and a repeated control. All use identical compact-root/early-operand
arithmetic and eight requested copy workers. CPU instructions have feature
checks and a memcpy fallback. Each mode creates and destroys its own State;
no policy changes while workers or GPU work are outstanding.

Readiness gates load a naturally aligned mapped host flag with a system-scope
acquire. Each worker fences its non-temporal stores before the last worker
publishes readiness. Every dependent DMA follows its gate; all GPU work is
joined before return. No GPU work or operand access occurs between calls.
No clocks or machine settings change. The whole graph optionally frees the
caller to pack its own disjoint rows, alongside the existing copy workers.

Three warm/20ms-idle/warm rounds per mode, with two verified warm calls before
each measurement. Mode order rotates across outer calls. Every result is
compared, and the unchanged harness independently verifies the returned answer.
These inner timings reuse the current inputs and exclude setup; they are not
standalone or ranked scores. Logs are emitted only after all measurements.

Memory ordering reference:
https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html
