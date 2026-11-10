# Physical-core CPU pipeline and shared twiddles

This candidate follows the official queue-window diagnostic:

- Eight copy workers keep only two input strips ahead of GPU submission.
- AVX2 streaming stores avoid destination read-for-ownership traffic. Each
  worker fences once per job before publishing completion; ordinary copies
  remain available when AVX2 is absent or a row is small.
- On systems exposing at least ten distinct physical cores, helper threads use
  separate cores. The caller's affinity is never changed. Smaller or unsupported
  systems retain normal scheduling. Placement failures fall back safely.
- The 512-point middle transform retains operand values in registers and shares
  a compact, swizzled twiddle table across its four warps. The table is generated
  only from public transform parameters. SM120 register use fell from 80 to 56
  without spills; one initial barrier loads the shared table.

The early CRT padding initialization, direct input conversion, paired transfers,
exact reconstruction, and full output scattering remain inside the timed call.
There is no GPU work between calls or reuse of earlier answers.

This exact source passed all 138 independent local cases. The streaming queue
passed 15,360 jobs over six team sizes. A separate affinity test confirmed that
an owned helper was placed correctly and the caller's affinity stayed unchanged.
The eight-core local machine uses the unbound fallback for the full candidate;
its binding helper was exercised independently. SM120 compilation passed with
zero spills. Local main-case runtime was about 9.08 ms, not an official score.

FHERMA artifact path: `candidates/physical-core-pipeline`.
