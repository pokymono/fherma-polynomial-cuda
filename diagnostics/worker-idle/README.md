# Worker idle and operand pipeline diagnostic

Unscored comparison of the existing UMONITOR/UMWAIT worker wait with ordinary
PAUSE polling, using both baseline and experimental pipelines. Workers poll only
their job-generation counter; no GPU work or operand access occurs between calls.
No machine settings, clock locks, harness allocations or benchmark rules change.
Polling consumes more CPU while idle, which is a practical tradeoff to measure.
The mode switches only after all threads from the previous mode have joined.

The new operand pipeline stages A first and transforms it while B is uploading.
Eight 64-column input strips per operand retain the old transfer size. Output
keeps sixteen 32-column strips. Existing exact modular operations and roots are
used in separate forward-A and forward-B/product/inverse kernels.

Each mode measures three rounds of warm/20ms-idle/warm calls, each following two
verified warm calls. All work is inside the unchanged outer harness clock;
reported inner timings alone compare the variants. Baseline modes bracket the
experiment. Logs are emitted compactly after all measurements. Every result is
compared and the returned answer is independently checked by the harness. Inputs
are reused within the current call only, so these are not standalone scores.

Official unscored run `6af378ad698e99b16a7ceaf5` passed 3/3 cases; all
162 measured inner samples were retained. Baseline warm brackets were
1.037087/1.037344 ms. The operand pipeline measured 1.022333 ms with normal
worker waits and 1.019922 ms with PAUSE polling. Baseline PAUSE polling was
1.039391 ms and did not remove delayed outliers. The tuned/polling mode had no
>1.2 ms delayed samples in nine trials, but this experiment lacks a simultaneous
tuned/normal-wait control and idle behaviour varies across runs. It does not
establish that polling fixes the idle penalty. No ranked candidate was promoted.
