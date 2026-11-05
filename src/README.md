# FHERMA CUDA implementation

This directory is the runnable source artifact for
`polynomial-multiplication/negacyclic@1.0.0`.

- Harness language: `cpp`
- Execution image: `cupqc`
- Entry points: `fherma_init`, `fherma_run`, `fherma_free`
- Build: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`

`solve.cu` implements exact RNS/NTT multiplication and integer tensor-core
conversion. `big_unsigned.h` provides host arithmetic for parameter-only setup.
`main.cpp` and `fherma.h` are the unchanged generated local harness and interface.

See the repository README for the arithmetic, testing, and timing boundaries.
