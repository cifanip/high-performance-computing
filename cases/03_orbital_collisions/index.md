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
\ddot{\mathbf{r}} = -\frac{\mu}{r^3}\mathbf{r} + \mathbf{a}_{J_2}, \qquad (1)
$$

where $\mu$ is the standard gravitational parameter and $\mathbf{a}_{J_2}$ the oblateness perturbation

$$
\mathbf{a}_{J_2} = -\frac{3}{2} J_2 \left( \frac{\mu R_E^2}{r^5} \right) \left[ \left( 1 - 5\left(\frac{z}{r}\right)^2 \right) \mathbf{r} + 2z \hat{\mathbf{k}} \right],
$$

with $R_E$ the Earth's radius and $J_2$ a constant that quantifies the primary oblateness of the planet. Given an initial condition $\mathbf{X}(t_0)=[\mathbf{r}(t_0),\dot{\mathbf{r}(t_0)}]$, equation (1) can be numerically integrated using an ODE solver, such as the **RK4 method** employed in the subsequent sections. However, observations inherently carry uncertainty due to measurement errors. Furthermore, nominal predictions of conjunction events between two objects, $\mathbf{X}$ and $\mathbf{Y}$, are typically affected by modeling errors introduced by the approximate analytical methods used for broad orbital screening. Consequently, the collision problem is fundamentally probabilistic, yielding the probability of collision ($P_c$) as the primary metric of interest.

To estimate $P_c$, we introduce a **Monte Carlo simulation** framework. In particular, we simulate an ensamble of perturbed initial conditions from a given associated covariance matrix $\mathbf{Q} \in \mathbb{R}^{6\times 6}$ in a neightborhood of the collision site. Each object pair is evolved by a high-resolutuon RK4 integrator and physical collisions are detected along their paths. 

### Algorithm Outline
We are given a list of $C$ collision sites containing state vector pairs of orbital objects $\mathbf{X}(T)$, $\mathbf{Y}(T)$, with $T$ the time of closest approach (TCA). To set up the Monte Carlo sampling, we first back-propagate the sate vectors to an earlier time $t_0 = TCA - \Delta t$. We then generate an ensamble of $N$ realizations for each space object and integrate them forward to $t=TCA$. Physical collisions are detected and recorded across all trajectory pairs, from which a discrete probability of collision $P_c$ is computed. 

For our numerical experiments, we employ the following simulation parameters: $C=500$ collision sites; $N=10^4$ realizations; $\Delta t = 12$ minutes. Furthermore, the space objects are assumed to be resident in Low Earth Orbit (LEO). Consequently, we consider a mean orbital radius on the order of $r \approx 7000$ km and an associated orbital velocity of $v \approx 7.5$ km/s.

## GPU-accelerated implementation
In the following sections, we outline several key implementation aspects to consider for achieving excellent computational performance on a GPU. 

### Memory layout
Even though memory bandwidth is not the primary concern in the RK4 method &mdash; which is notoriously compute-bound &mdash; an efficient memory layout is essential to avoid data read/write bottlenecks. Suppose input sate vectors arrive from an observational system as an ordered list:

$$
\lbrace \mathbf{X}^0(T),\mathbf{X}^1(T),...,\mathbf{X}^{2C-1}(T) \rbrace, \qquad (2)
$$

with the coordinates of an element in the list given by

$$
\mathbf{X}^i(T) = [ \mathbf{x}^i(T),\mathbf{y}^i(T),\mathbf{z}^i(T),\mathbf{v_x}^i(T),\mathbf{v_y}^i(T),\mathbf{v_z}^i(T) ].
$$

List (2) must be expanded to generate ensambles for the Monte Carlo simulation. Storing coordinates as an interleaved sequence $x,y,z,...,v_z$ as in (2) creates an inefficient layout. Threads within a warp accessing these coordinates from VRAM would fetch memory strided by 6 elements, resulting in uncoalesced accesses and wasted cache lines. 

To achieve strict **memory coalescence** we construct a layout by storing values of the same coordinate sequentially. For a single object, e.g. $X^0$, we have

$$
L^0 = \lbrace \mathbf{x}^0_0,...,\mathbf{x}^0_{N-1},\mathbf{y}^0_0,...,\mathbf{y}^0_{N-1},...,\mathbf{z}^0_{0},...,\mathbf{z}^0_{N-1},... \rbrace,
$$

where the subscript indicates the realization. The full dataset is obtained by concatenating the lists $L^i$:

$$
L = \lbrace L^0, L^1, L^{2C-1} \rbrace. \qquad (3)
$$

Consequently, when the threads in a warp execute an RK4 iteration, consecutive threads fetch consecutive coordinate values. This contiguous alignment guarantees coalesced memory transactions, maximizing VRAM bandwidth utilization.

### Random sampling
Random sampling of the initial conditions is carried out when expanding the input state vector list (2). First, a one-time initilization of `curandState` objects is executed in a CUDA kernel:

```
curand_init(seed, thread_idx, 0, &curand_state[thread_idx]);
```

where `thread_idx` is the global thread index, ensuring each thread is associated with a unique random state. To generate a random number, a thread must access its associated state from VRAM. Since the state vector is 6-dimensional, multiple samples have to be drawn. To avoid repeated accesses to the VRAM, the random state is copied to thread-local memory so that subsequent samples are performed efficiently directly from registers. Here is an example of applying a uniform and independent perturbation to the position coordinates:

```
  // local copy
  curandState local_state = rand_states[thread_idx];
  
  // position perturbation
  double dxA = curand_uniform_double(&local_state) * 0.2 - 0.1;
  double dyA = curand_uniform_double(&local_state) * 0.2 - 0.1;
  double dzA = curand_uniform_double(&local_state) * 0.2 - 0.1;
```

### Occupancy
The wrokload is organized in a two-dimensional grid, where the first dimension spans $N$ realizations while the second dimension spans $C$ collision sites. Given a warp has $32$ threads, and the coalsced memory reads (see previous section), the block dimension along $x$ should be at least $32$. We conduct numerical experiments on a **NVIDIA Testla V100**. The maximum number of blocks per SM is 32, while the maximum number of threads per block is 1024 with an overall maximum number fo threads per SM of 2028.

The workload is organized into a two-dimensional grid, where the first dimension ($x$) spans the $N$ realizations and the second dimension ($y$) spans the $C$ collision sites. Because a warp consists of 32 threads &mdash; and to preserve the coalesced memory reads detailed in the previous section &mdash; the block dimension along $x$ must be a multiple of 32. We conduct our numerical experiments on an NVIDIA Tesla V100. For this architecture, the maximum number of blocks per Streaming Multiprocessor (SM) is 32, the maximum number of threads per block is 1024, and the absolute maximum number of threads per SM is 2048. 

The most stringent costraint on achieving maximum occupancy is the number of registers available per thread, in this case equal to $65,536 / 2048 = 32$ registers. An insection of the compilation output  (using `nvcc -arch=sm_70 --ptxas-options=-v`) reveals that the main RK4 CUDA kernel requires $85$ registers:

```
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 85 registers, 408 bytes cmem[0], 24 bytes cmem[2]
```

This results in about $38$% occupancy. One could, for example, force a kernel of $256$ threads per block to reach $50$% occupancy by requesting at least $4$ blocks

```
__global__ void __launch_bounds__(256, 4) rk4(...)
```
However, this introduces a tradeoff between the gain in occupancy and the latency penalty of register spilling to local memory. The table below compares the execution times for both configurations. In practice, accepting the register limit and avoiding spilling proves to be the preferable choice. Similar tests yielded analogous results.

| Configuration | Execution Time (ms) |
| :--- | :--- |
| Low occupancy - no spill overs | 522.31 |
| High occupancy - spill overs | 547.61 |

### Collision count reduction
Computation of the probability $P_c$ requires counting the number of collisions that occur within the ensemble of trajectories. Since multiple threads must update this count for the same collision site, memory races become an issue. A simple remedy is to use atomic additions

```
if (collided)
{
  atomicAdd(&counter[cid], 1);
}
```

where a shared counter is updated every time a atomically collision is detected. However, not all threads need to call `atomicAdd`. Instead, threads within a warp can cooperate to aggregate the local count, allowing a single thread to perform the global atomic sum. This reduces accesses to global VRAM by a factor of 32. An sketch of this implementation is provided below:

```
  // 32-bit mask
  unsigned int collision_mask = __ballot_sync(0xFFFFFFFF, collided);
  
  int lane_id = threadIdx.x % 32;
  
  // thread leader in the warp
  if (lane_id == 0) 
  {
    // Population count
    int collisions_in_warp = __popc(collision_mask);
    
    // atomic reduction on RAM
    if (collisions_in_warp > 0) 
    {
      atomicAdd(&counter[cid], collisions_in_warp);
    }
  }
```
In practice, however, since this algorithm is heavily compute-bound, this memory optimization yields a negligible overall speedup.

### FLOPs count and performance measurament




