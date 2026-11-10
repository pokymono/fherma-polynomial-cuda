# Register NTT candidate

This candidate retains the exact independent-prime RNS construction and CRT
correction from `src/`. It changes three GPU execution details:

- Input conversion and CRT use separate integer tensor-core tile sizes, with
  shared-memory swizzling and contiguous output stores.
- The six smallest middle-transform stages use registers and warp shuffles.
- Upper row stages use statically indexed registers, leaving only three stages
  that exchange values between warps through shared memory.

All original generic fallbacks remain. The generated harness is unchanged.

On November 10, 2026, this source passed 138 independent exact local cases,
including both challenge sizes and wider/larger fallback cases. CUDA compilation
for SM120 succeeded with no register spills in the changed kernels.

Alternating local runs on an RTX 3050 Laptop measured about 14.62 ms versus
15.96 ms for the baseline at `(65536,1747)`, a 1.092x speedup. These are local
development measurements, not official Blackwell results. No leaderboard result
is claimed for this candidate until the platform measures this exact source.
