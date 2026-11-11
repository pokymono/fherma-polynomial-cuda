# CPU copy-core placement diagnostic

Four layouts rotate across fresh supplied calls: existing first cores (0), last
available cores (1), alternating beginning/end cores (2), and evenly spaced cores
(3). The caller and CUDA dispatcher keep their existing cores. Alternative sets
exclude those two cores and stay within the process's original allowed affinity.
If too few cores are available, the existing placement remains in use.

Every mode reapplies its copy-worker affinity inside the measured multiplication,
including the baseline. A promising layout must be confirmed in a standalone
candidate with setup-time placement. No machine settings, input allocations,
CUDA dependencies, or exact arithmetic change. Each input is multiplied once;
all workers and GPU work finish before return. Logs record the pre-log total and
chosen cores. These diagnostic timings are not ranked candidate scores.
