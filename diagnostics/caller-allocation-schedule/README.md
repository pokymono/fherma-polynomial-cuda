# Fresh-call allocation schedule diagnostic

Each supplied input is multiplied once, with four public allocation schedules
rotated across calls: immediate construction (mode 0), after input strip 3 is
packed (mode 1), after operand A is packed (mode 2), and after both operands are
packed (mode 3). The GPU dispatcher runs concurrently throughout. These waits
use existing CPU publication flags and add no GPU event or GPU synchronization.
Output ownership, exact arithmetic, and the generated harness are unchanged.

CPU timestamps are logged in a single line shorter than 300 characters. The
pre-log total excludes formatting; official outer timings include it. Rotation
is a diagnostic comparison, not a standalone candidate or ranked submission.
Every worker and GPU operation finishes before returning. No work persists
between calls and no answers or inputs are cached.

Official run `6af3da53698e99b16a81c053` passed 20/20. Five scored calls per mode showed no convincing gain from delaying allocation. See RESULTS.json for complete compact traces; no delayed mode was selected.
