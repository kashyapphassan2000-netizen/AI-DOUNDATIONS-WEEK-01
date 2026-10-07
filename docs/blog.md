# I built tiny NumPy and learned why NumPy is fast

I spent a week rebuilding the core of numerical Python from scratch: a `Vector`, a `Matrix`, a scalar reverse-mode autograd engine, a small neural net, and SVD/PCA. No NumPy inside the library. NumPy only appears in tests, as the answer key.

**What I built.** `Vector` and `Matrix` with dot, norm, transpose and two matrix multiplies (the triple loop and Strassen). A `Value` class that records how each number was produced and runs the chain rule backwards over a topologically sorted graph. A `Neuron`/`Layer`/`MLP` on top of it, trained on XOR to near-zero loss. SVD by power iteration with deflation, used for PCA on iris. Every piece is checked against NumPy or scikit-learn, plus a hypothesis property test and a finite-difference gradient check.

**The speed gap.** A 128x128 multiply took 0.107 s in my code and 0.00047 s in NumPy: about 229x slower. Same math. NumPy hands the work to optimized C/BLAS over contiguous memory; pure Python pays interpreter overhead on every single multiply-add. That number is the real answer to "why is NumPy fast".

**The Strassen surprise.** Strassen does 7 sub-multiplies instead of 8, so it should win on large matrices. In pure Python it lost at n=128 and n=256 (0.137 vs 0.121 s, 1.046 vs 0.950 s) and only won at n=512 (7.27 vs 10.74 s). The extra additions and list slicing cost more in an interpreter than the saved multiply, so the crossover sits far above the textbook number. Asymptotics tell you the shape of the curve, not where the lines cross.

Code: https://github.com/kashyapphassan2000-netizen/AI-DOUNDATIONS-WEEK-01
