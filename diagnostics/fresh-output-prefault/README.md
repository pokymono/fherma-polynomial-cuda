# Fresh output population diagnostic

Each harness call executes exactly one ordinary multiplication on its fresh
inputs. Modes rotate across calls: ordinary output resize (0), parallel
MADV_POPULATE_WRITE on reserved output capacity (2). All reservation, page population, initialization
and copies occur inside the measured call. Existing input-copy workers help
populate disjoint page ranges after finishing their input copies.

Only fully owned interior pages of the current output allocation receive
advice. The ordinary vector still performs its standard resize and zero
initialization. No global allocator or machine settings change. Advice errors
are counted and ordinary resize remains the correctness fallback. Every
output goes to the independent harness verifier; there are no inner repeats,
extra warmups, sleeps or retained outputs. This is an unscored diagnostic;
CPU phase observations and outer times including logging are not candidate scores.
