# Fresh-call worker-count diagnostic

One persistent team is constructed in setup, requesting ten copy workers within
the allowed physical-core limit. Each supplied input is multiplied once, with
four, six, eight, or ten active workers in rotation (capped by available workers).
Caller and dispatcher placement stays fixed across all modes. Inactive workers
return immediately; active workers own disjoint input/output row intervals.
The caller handles the final output interval and joins every worker before return.

This isolates worker count from caller placement and avoids repeated-input warm
measurements. No threads are created during a measured call. Every mode still
wakes the full team, so a promising count needs standalone confirmation with the
smaller persistent team. No CUDA work or exact arithmetic changes. Compact logs
include pre-log total time; official timings include logging. Not a ranked score.
