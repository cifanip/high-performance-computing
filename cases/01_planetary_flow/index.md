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
\dot{\omega} = \lbrace \psi, \omega \rbrace, \qquad (1.0) \\
\Delta \psi = \omega,
\end{cases}
$$

where $\psi$ is the stream-function, related to vorticity $\omega$ via the Laplace-Beltrami operator $\Delta$, and $\lbrace \cdot,\cdot \rbrace$ is the Poisson bracket. The construction of a numerical scheme that embeds a Lie-Poisson structure and its conservation laws requires a finite truncation of the Poisson bracket. This subject relates to concepts of differential geometry, for which I refer the reader to [^1]. In brief, there exists a rigorous procedure to approximate smooth functions on the sphere by finite-dimensional matrices in the unitary group $\mathfrak{u}(N)$. By doing so, one obtains a spatial discretization of (1.0) in the matrix form

$$
\begin{cases}
\dot{W} = [P, W], \qquad (1.1) \\
\Delta_N P = W,
\end{cases}
$$

where $W \in \mathfrak{su}(N)$ is the vorticity matrix, $P \in \mathfrak{su}(N)$ is the stream matrix and $\Delta_N$ a discrete Laplacian. System (1.1) preserves $\text{Tr}(W^k)$ for $k=1,...,N$, which corresponds to the integrated powers of vorticity in the continuum. 

From a numerical standpoint, the three main components are:

* the computation of matrix multiplications stemming from the commutator,
* the construction of the Lie-algebra basis,
* the computation of the inverse Laplacian.

The matrix multipliucation will dominate the computatinal complexity being $\mathcal{O}(N^3)$. The other two points require the solution of eigenvalue problem and of a linear system. As it stands, the problem is prohibitevly expensive, since the Laplacian matrix is a forth-order tensor. However, $\Delta_N$ admits a tridiagonal splitting into $(2N-1)$ block of size $N-|m|$ corresponding to the $m$th diagonal of $W$. The parallel implementation of these algorithms will be discussed in the following.

## 2. Parallel CPU implementation




[1]: Cifani, P., Viviani, M. and Modin, K., 2023. An efficient geometric method for incompressible hydrodynamics on the sphere. Journal of Computational Physics.

[2]: Franken, A.D., Caliaro, M., Cifani, P. and Geurts, B.J., 2024. Zeitlin truncation of a shallow water quasi‐geostrophic model for planetary flow. Journal of Advances in Modeling Earth Systems.
