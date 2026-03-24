# Extended Kalman Filter for Orbital State Estimation

This project implements an Extended Kalman Filter for estimating the 2D orbital state of a satellite from bearing-only measurements.

## Overview

The satellite dynamics are nonlinear due to gravitational acceleration, and the measurement model is nonlinear because the sensor provides only the bearing angle from the origin to the satellite. The filter estimates position and velocity over time using recursive propagation and local linearization.

## Features

- Nonlinear orbital dynamics under central gravity
- Bearing-only measurement model
- Extended Kalman Filter update and propagation
- Analytical Jacobians for both dynamics and measurement models
- Innovation angle wrapping for stable filtering
- Error analysis with 3-sigma covariance bounds

## How to Run

Run the main script:

```bash
python src/extended_kalman_filter.py
```

## Report

See the full writeup and results in:

```report/ekf.pdf```