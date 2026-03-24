# Particle Filter Localization with Lidar

This project implements a particle filter for robot pose estimation in a mapped indoor environment using noisy control inputs and lidar range measurements.

## Overview

The robot state consists of 2D position and heading. A set of weighted particles is used to represent the state distribution, and the filter recursively performs:
- measurement update using a nonlinear lidar measurement model
- state estimation via weighted averaging
- resampling based on the effective number of particles
- propagation through nonlinear robot dynamics with process noise

## Features

- Nonlinear state and measurement models
- Particle-based representation of uncertainty
- Weighted state and covariance estimation
- Resampling based on particle degeneracy
- Localization under varying levels of initial uncertainty
- Visualization of trajectories, estimation errors, and particle behavior

## Cases Studied

The particle filter is evaluated under four initialization conditions:
- known position, known heading
- unknown position, known heading
- known position, unknown heading
- unknown position, unkown heading (cold-start)

These cases illustrate how particle count requirements increase with prior uncertainty.

## How to Run

Run the main script:

```bash
python src/particle_filter.py
```

## Simulation

Run the simulation:

```ani.svg``` 

to see the robot and particle behavior.

## Report

See the full writeup and results in:

```report/particle_filter.pdf```