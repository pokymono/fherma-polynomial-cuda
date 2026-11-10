Pinned host staging for the same exact RNS implementation in `src/`.

Only parameter-sized empty buffers are allocated in initialization. Every call
copies the current operands into pinned memory, transfers them to the GPU,
computes the complete polynomial product, transfers the answer back, and copies
it into the returned vector before returning. No operand or output is cached.

The generated harness files are unchanged. This candidate passed 24 independent
exact cases at each competition point locally before official benchmarking.
