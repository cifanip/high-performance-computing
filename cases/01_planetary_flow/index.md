---
title: Planetary Flow
layout: default
---

# Planetary Flow

This section summarizes an efficient and highly-scalable solver developed for the simulation of **geophysical flows** on a spherical surface (see [1] and [2] for details). The original implemention presented in [1] was based on an hybrid MPI-OpenMP parallelization. More recently, I have developed a **GPU-accelerated** version to conduct research in the group of prof. Klas Modin at Chalmers University. A 10x speedup was achieved compared to the already massively parallelized CPU version. 

## 1. Problem setup

Two-dimensional hydrodynamics possess geometric properties that affect its qualitative long-time behaviour. In particular, there exist infinitely many first integrals, called Casimir functions. Physically, they state the conservation of the integrated powers of vorticity and reflect that vorticity is advected along stream-lines. There is strong evidence that the long-time qualitative dynamics of non-viscous two-dimensional fluids is tied to the conservation of Casimirs. To better capture the correct long-time behaviour it is therefore natural to construct numerical methods that preserve these conservation laws. 

Consider, as a prototype model, the Euler flow on the sphere governed by the euqations

$$
\begin{cases}
\dot{\omega} = \{\psi, \omega\}, \qquad (1.0) \\
\Delta \psi = \omega,
\end{cases}
$$


[1]: Cifani, P., Viviani, M. and Modin, K., 2023. An efficient geometric method for incompressible hydrodynamics on the sphere. Journal of Computational Physics.

[2]: Franken, A.D., Caliaro, M., Cifani, P. and Geurts, B.J., 2024. Zeitlin truncation of a shallow water quasi‐geostrophic model for planetary flow. Journal of Advances in Modeling Earth Systems.
