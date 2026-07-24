---
title: Orbital Collision Simulator
layout: default
---

# Orbital Collision Simulator

This section details a **GPU-accelerated** simulator of **orbital collisions**. Memory layout, register pressure and computational perfomance are illustreated. 

Parallel Programming Stack:
* CUDA
* cuRAND

## 1. Problem setup

The orbital collision problem can be stated in classical mechanics as an initial value problem. The objective is to determine whether the trajectories of a pair of spatial objects will intersect in a given time horizon. For this purpose, point-particle dynamics serves as an appropriate model. Let us assume that the satellite's position vector is governed by the following second-order differential equation

$$
\ddot{\mathbf{r}} = -\frac{\mu}{r^3}\mathbf{r} + \mathbf{a}_{J_2} \qquad (1)
$$

where $\mu$ is the standard gravitational parameter and $\mathbf{a}_{J_2}$ the oblateness perturbation

$$
\mathbf{a}_{J_2} = -\frac{3}{2} J_2 \left( \frac{\mu R_E^2}{r^5} \right) \left[ \left( 1 - 5\left(\frac{z}{r}\right)^2 \right) \mathbf{r} + 2z \hat{\mathbf{k}} \right]
$$

with $R_E$ the Earth's radius and $J_2$ a constant that quantifies the primary oblateness of the planet. Given an initial condition $\mathbf{x}(t_0)=[\mathbf{r}(t_0),\dot{\mathbf{r}(t_0)}]$, equation (1) can be numerically integrated using an ODE solver, such as the **RK4 method** employed in the subsequent sections. However, observations inherently carry uncertainty due to measurement errors. Furthermore, nominal predictions of conjunction events between two objects, $\mathbf{x}_0$ and $\mathbf{x}_1$, are typically affected by modeling errors introduced by the approximate analytical methods used for broad orbital screening. Consequently, the collision problem is fundamentally probabilistic, yielding the probability of collision ($P_c$) as the primary metric of interest.

To estimate $P_c$, we introduce a **Monte Carlo simulation** framework. In particular, we simulate an ensamble of perturbed initial conditions from a given associated covariance matrix $\mathbf{C} \in \mathbb{R}^{6\times 6}$ in a neightborhood of the collision site. Each object pair is evolved by a high-resolutuon RK4 integrator and physical collisions are detected along their paths. 

### Algorithm Outline
We are given a list $i=0,...,C-1$ of collision sites containing state vector pairs of orbital objects $\mathbf{x}(t^*)_0^i$, $\mathbf{x}(t^*)_1^i$, with $t^*$ the time of closest approach (TCA).  




