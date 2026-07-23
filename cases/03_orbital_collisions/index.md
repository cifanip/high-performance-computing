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

The orbital collision problem can be stated in classical mechanics as an initial value problem. The objective is to determine whether the trajectories of a pair of spatial objects will intersect in a given time horizon. 
