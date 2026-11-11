# Fresh output construction placement diagnostic

Exactly one normal multiplication runs per fresh harness call. Four modes
rotate: the existing output worker, the caller after queuing GPU work, the
first copy worker after its input copies, and the output worker on the first
copy worker's CPU sibling. All allocation and initialization remain timed.
The ordinary vector allocator, resize, ownership and output contract remain
unchanged. No machine or global allocator settings are changed.

CPU topology is read at initialization. Switching the output worker's CPU
between modes is timed; a standalone fixed-placement candidate would pin
once at initialization. The diagnostic records allocation wall time and
thread page-fault deltas. User/system CPU times have coarse accounting
resolution and a zero does not mean no CPU work occurred.

Every fresh output goes to the independent verifier. There are no inner
repeats, retained answers, extra warmups or sleeps. This is an unscored
diagnostic, not a candidate score. Outer timings include diagnostic logging.

Official run `6af3c0c4698e99b16a806b08` passed 20/20. Five measured caller-construction cases had a median 1.058058 ms; the other placements were slower. Mixed allocation sites and timed worker affinity changes limit this comparison; standalone fixed candidates are required. See RESULTS.json.
