# Exact direct Montgomery conversion of byte dot products

Let R = 2^32 and let p be an odd NTT prime with 2^30 <= p < 2^31.
Initialization enforces this prime range. The tensor path handles at most
32768 byte terms per dot product, so each nonnegative dot is at most
32768 * 255^2 = 2,130,739,200 < 2^31. Signed INT32 MMA accumulation is exact.

Folding four such dots at byte offsets 0, 8, 16, and 24 gives a nonnegative
64-bit integer S < 2^55. In particular, floor(S/R) < 2^23 < p, so S < pR.
Let m = (S mod R) * p^(-1) mod R. Then S-mp is divisible by R, and

```
v = (S - m*p)/R
```

lies strictly between -p and p. Add p exactly when v is negative. The result
is the canonical representative of S*R^(-1) modulo p. The CUDA implementation
computes this with the high words of S and m*p and an explicit borrow check.
It uses integer arithmetic only.

The tensor path precomputes its negacyclic twist as psi^i*R^3 modulo p.
One ordinary Montgomery product then gives

```
REDC(v * (psi^i*R^3)) = S * psi^i * R mod p,
```

the same twisted Montgomery representation required by the NTT. Scalar input
conversion retains its original R^2 twist because it starts with an ordinary
canonical residue. Initialization and dispatch use the same public shape
predicate to choose these two scales.

`tests/tensor_montgomery_math.py` checks the identities using independent
arbitrary-precision modular arithmetic, including zero, modulus boundaries,
the maximal byte-sum bound, and values near pR. The full CUDA suite separately
checks end-to-end polynomial outputs against the independent GMP oracle.
