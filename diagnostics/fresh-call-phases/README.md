# Fresh-call phase diagnostic

One ordinary multiplication executes per harness call. The source records
CPU timestamps around input dispatch, output allocation and output event
waits. Output allocation and its worker wake delay are measured separately.
There are no extra inner warmups, repeated inputs, sleeps or added GPU events.
Every returned output goes to the normal independent verifier.

The `total` field stops before diagnostic logging. Official outer times include
that logging and are not candidate scores. CPU event-wait timestamps show when
the caller observes completion; they are not direct GPU execution timestamps.
This diagnostic is for distinguishing host allocation/dispatch delay from a
late GPU pipeline under the official fresh-input harness.
