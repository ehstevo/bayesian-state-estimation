# Linear Kalman Filter for 2D Motion Tracking

This project implements a discrete-time linear Kalman Filter for estimating 2D position and velocity from acceleration inputs and noisy position measurements.

## Overview

The system is modeled in continuous-time state-space form and discretized using the van Loan method. Measurements are taken in a rotated sensor frame, creating coupling between the position states. A Kalman filter is then used to  recursively estimate the full state and covariance over time.

## Features

- Linear continuous-time state-space model
- Exact discrete-time conversion using the van Loan method
- Recursive Kalman Filter update and propagation
- Simulation of truth, inputs, and noisy measurements
- Comparison of predicted uncertainty with observed estimation error
- Analysis of covariance structure under different sensor configurations

## How to Run

Generate the simulated data:

```bash
python src/util.py
```

Run the Kalman Filter:

```bash
python src/linear_kalman_filter.py
```

## Report

See the full writeup and analysis in:

```report/linear_kalman_filter.pdf```