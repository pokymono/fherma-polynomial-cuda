# Caller pipeline CPU timestamps

This diagnostic runs the checked four-worker candidate once per supplied input.
It records CPU timestamps around existing input publication, GPU dispatch,
ordinary output construction, output event waits and host scatter. It adds no
CUDA event, synchronization, repeated multiplication, or between-call GPU work.
All workers still finish before return. Timestamps and stderr logging add
measurement overhead, so its outer time is not a standalone candidate score.
The log total is sampled before formatting and printing; it also is not the
official outer duration. Input data and the generated harness are unchanged.

Official run `6af3d70b698e99b16a819481` passed 20/20. Its outer median was 1.2008405 ms. The platform truncated each log line after 300 characters, losing the output timestamps. RESULTS.json retains only complete fields, paired to commit `c427981744f8a8685418298980d712aee25a0dda`.
