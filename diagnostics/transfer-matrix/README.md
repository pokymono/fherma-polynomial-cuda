# Transfer comparison diagnostic

This artifact compares five complete execution paths using the register-NTT
candidate's arithmetic: pageable copies, pinned staging with direct vector
construction, parallel pinned staging, GPU access to mapped staging, and
per-call host registration. Each path reads the supplied operands and computes
the entire answer anew inside `fherma_run`. Its wall-clock duration is printed
to stderr. Every answer is compared against the first path before returning.

Only empty, parameter-sized staging buffers and a bounded CPU worker team are
created in initialization. No operand-dependent work occurs between calls, and
all worker activity is joined before returning. Registrations are removed at
the end of each path. The generated harness is unchanged.

The platform's reported time includes all five computations, comparisons, and
logging. This is a diagnostic artifact, not a leaderboard candidate. The
individual timings are for selecting a later production transfer strategy.

The final diagnostic passed twelve independent local cases across the two
challenge sizes. Official hardware validation remains pending.
