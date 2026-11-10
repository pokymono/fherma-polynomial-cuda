# Streaming lane diagnostic

This is an unscored diagnostic of the ordered-streaming implementation. It
compares 3, 4, 6, 8, and 10 requested CPU input workers, bounded by the allowed
physical-core count. Each worker configuration reserves a caller core and, when
available, a separate allocation core. It tests 8 workers at both ends to expose
drift. Every mode computes the full exact answer; outputs are compared.

All thread creation, arithmetic, allocation, comparisons, and logs occur inside
the official timed call. Printed inner timings are diagnostic measurements,
not leaderboard scores. The generated harness is unchanged. Use only the
private profile implementation with three verified cases, never the challenge
submission endpoint.
