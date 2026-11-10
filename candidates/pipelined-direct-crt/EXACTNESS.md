# Exact shortened CRT reconstruction

Let `N` be the polynomial length and `q` the supplied coefficient modulus.
For canonical input coefficients, every signed coefficient `x` of their
negacyclic integer product satisfies `|x| <= B = N (q-1)^2`.

Choose distinct 31-bit NTT primes `p_i` with product `P > 4B`, and define
`P_i = P/p_i`. The inverse transform produces the normalized residues
`v_i = x * inverse(P_i) mod p_i`, each in `[0,p_i)`.
Consequently:

```
sum(v_i P_i) = x + K P
sum(v_i / p_i) = K + x/P
```

Because `|x/P| < 1/4`, rounding the rational sum to the nearest integer gives
the exact integer `K`, including when `x` is negative.

## Integer fixed-point quotient

Precompute `r_i = floor(2^64/p_i)`. Since each prime is odd,
`UINT64_MAX / p_i` is exactly this value. Compute

```
t_i = (uint64(v_i) * r_i) >> 32
K = (sum(t_i) + 2^31) >> 32
```

The product fits in 64 bits because `v_i < p_i`. The error in each term is
nonnegative and strictly less than `3/2` in units of `2^-32`:
truncating the reciprocal contributes less than `p_i/2^32 < 1/2`,
and the final shift contributes less than one. For fewer than 8192 primes,
the total error is less than `3*8192/2^33`, much smaller than the quarter-integer
rounding margin. The integer expression therefore returns the exact `K`.
The code explicitly checks the prime-count bound before selecting this path.

## Shortened reconstruction

Precompute `M_i = P_i mod q` and `R = (-P) mod q`. Then

```
T = sum(v_i M_i) + K R >= 0
T mod q = x mod q
```

`0 <= K <= r`, where `r` is the number of primes. Thus
`T < r * 2^31 * q + r*q < 2^(W+45)` for `r < 8192` and `q < 2^W`.
The allocated `ceil(W/32)+2` words suffice. Tensor cores compute the byte dot
products exactly; their accumulation length is at most 32768 bytes and
`32768*255*255 < 2^31`. Word carries and the quotient correction use integer
operations.

For `q = 2^W - gap` with a nonzero 32-bit `gap`, folding the high bits by
`2^W = gap (mod q)` and then subtracting `q` produces the canonical answer.
The fast path requires at most 64 working words. Other moduli and sizes use
the full-width reconstruction fallback.

## Subtractive Montgomery multiplication

For `a,b < p < 2^31`, set `R=2^32` and
`m = (a*b mod R) * inverse(p) mod R`. The low words of `a*b` and `m*p`
are equal, so their high-word difference is exactly `(a*b-m*p)/R`.
It lies strictly between `-p` and `p`. Adding `p` when the unsigned high-word
comparison detects a negative difference produces the canonical Montgomery
product. No signed overflow or floating-point approximation is required.

## Independent checks

`tests/direct_crt_math.py` checks this derivation with Python arbitrary-precision
integers at four parameter points, including both competition sizes. It tests
extremal signed convolution bounds and 1000 random signed integers per point.
Full GPU outputs are separately checked against `tests/reference.py`, whose GMP
coefficient-packing oracle does not share the implementation's RNS or NTT.
