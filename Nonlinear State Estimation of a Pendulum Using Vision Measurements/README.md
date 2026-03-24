# Nonlinear State Estimation of a Pendulum Using Vision Measurements

This project estimates the full state of a pendulum system using nonlinear dynamics and camera-based measurements. Three Bayesian estimation methods are implemented and compared: an Extended Kalman Filter (EKF), a sigma-point Kalman filter (UKF), and a particle filter (PF).

## Overview

A ball suspended from a string is tracked using a monocular camera. The measurement model provides only image-space observations (pixel location and apparent radius), requiring the full 3D state to be inferred indirectly.

The system state includes:
- 3D position of the ball
- 3D velocity of the ball
- 3D position of the hook (unknown)

This results in a 9-dimensional nonlinear estimation problem with both geometric and physical constraints.

## System Model

The dynamics include:
- gravitational acceleration
- tension constraints (counter-gravitational + centripetal)
- aerodynamic drag

The measurement model is based on camera projection:
- pixel coordinates (u, v)
- apparent radius (depth-dependent)

This creates a highly nonlinear measurement relationship due to perspective effects.

## Estimation Methods

Three estimation approaches were implemented:

### Extended Kalman Filter (EKF)
- Linearizes dynamics and measurement models via Jacobians
- Efficient but sensitive to model nonlinearities

### Sigma-Point Kalman Filter (UKF)
- Uses deterministic sampling (sigma points)
- Better captures nonlinear transformations of uncertainty
- No analytical Jacobians required

### Particle Filter (PF)
- Represents the state distribution with weighted samples
- Handles non-Gaussian and multimodal uncertainty
- Computationally more expensive

## Results

All three filters were evaluated on real measurement data. Performance was assessed using:
- trajectory consistency
- measurement reconstruction accuracy
- intersection-over-union (IOU) metric in image space

The UKF provided the best balance between accuracy and stability, while the particle filter demonstrated robustness in highly nonlinear regions, and the EKF was the most computationally efficient.

## How to Run

Run each estimator independently:

```bash
python src/ekf_pendulum.py
python src/ukf_pendulum.py
python src/particle_filter.py
```

Each estimator will print out an average IOU metric and play an animation of the pendulum over time.

## Report

See full writeup and analysis in:

```report/pendulum.pdf```