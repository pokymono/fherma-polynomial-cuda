# Timed startup diagnostic

Compare immutable table prefetch and short arithmetic startup probes with a
no-probe control. Each probe begins inside the measured multiplication call,
overlaps its CPU preparation and input pipeline, and joins before return.
No background GPU work, clock setting, untimed warm-up, harness change,
input-specific setup or cross-call operand retention is introduced.

The table modes read forward roots, twists, both, or forward/inverse roots.
Arithmetic probe modes measure startup responsiveness with bounded FMA loops;
their cost is included in every reported inner time. These are experiments,
not selected optimizations or a claim about the cause of idle latency.

Every outer call executes 243 complete multiplications, compares every output,
and measures 81 inner calls with rotating mode order and repeated controls.
The ordinary harness independently checks the returned result. Warm and
20 ms-idle diagnostics are distinct from standalone fresh-call timing.

Official unscored run `6af3b527698e99b16a7fc103` passed 3/3. Warm controls
were 0.978716 and 0.978932 ms; 20 ms-idle controls were 1.401816 and
1.390443 ms. The longest compute probe gave 0.986509 ms warm and
1.285271 ms after idle. It reduced the observed idle gap but remained
well above the target; the cause is not established. The AWS proxy did
not reproduce the large idle penalty. Full samples are in RESULTS.json.
