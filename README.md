# qMatrixLib

### Introduction
This is a small and lightweight c++ template library for matrix algebra.

It provides simple matrix operations, linear solvers(LU, QR), eigen solvers(power method, QR algorithm) and some algorithms based on these, like inverse, null space and determinates.

### Limitations

Please note that this library is not inteded to be used for any high performance matrix algorithms. This is simply not the design goal of this library. It is inteded to be used as a small helper library (originally for CUDA) when developing algorithms (sort of prototyping in c++). Here are some limitations which should be considered

- All algorithms are implemented without optimizations of any kind, not even SSE is used.
- All algorithms are implemented directly following the mathemaical formulation, e.g. the LU decompositions use two matrices to store L and U while this is easily possible using half the storage space.
- All method results are returned on the stack which limits the matrix size dramatically!

### Installation
There should be nothing to do besides including the appropriate files.

### Bugs
- The `hessenbergReduction`function is not working correctly.
- The `svd` function is not working correctly.