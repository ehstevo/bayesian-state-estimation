# Sigma-Point Kalman Filter for Drone State Estimation

This project implements a Sigma-Point Kalman Filter (SPKF) for estimating the lateral motion of a drone with nonlinear dynamics. A square-root Unscented Kalman Filter (UKF) is used as the primary implementation for improved numerical stability.

## Overview

The drone state consists of:
- 2D position (px, py)
- speed (s)
- roll angle ($\phi$)
- heading angle ($\psi$)

The sytsem follows coordinated-turn dynamics, introducing nonlinear coupling between kinematic and attitude states. Measurements are limited to noisy GPS position, requiring the filter to infer the remaining states through system dynamics.

## Sigma-Point Filtering

Sigma-Point Kalman Filters approximate nonlinear transformations of a Gaussian distribution using a deterministic set of sample points (sigma points). These points are propagated through the nonlinear dynamics and recombined to recover the predicted meand and covariance.

This approach avoids the need for analytical Jacobians and provides improved accuracy over linearization-based methods such as the Extended Kalman Filter.

## Square-Root UKF Implementation

This project uses a **square-root Unscented Kalman Filter (UKF)**.

Instead of propagating the covariance matrix directly, the filter propagates its Cholesky factor. This provides:
- improved numerical stability
- preservation of positive definiteness
- more robust behavior over long simulations

The square-root formulation is particularly important for nonlinear systems where covariance matrices can degrade over time due to numerical error.

## Features

- Nonlinear coordinated-turn flight model
- Sigma-Point state estimation framework
- Square-Root UKF implementation
- Joint estimation of position, speed, roll, and heading
- Numerically stable covariance propagation
- Visualization of trajectory and estimation performance

## How to Run

```bash
python src/ukf.py
```

## Report

See the full writeup in:

```report/ukf.pdf```