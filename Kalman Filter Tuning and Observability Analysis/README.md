# Kalman Filter Tuning and Observability Analysis

This project studies the consistency of a linear Kalman Filter under covariance mismatch and analzes system observability using the observability matrix and singular value decomposition.

## Overview

A linear position-velocity estimation problem is used to examine how incorrect assumptions about process and measurement noise affect filter performance. The estimator is evaluated using three metrics:
- mean squared error (MSE)
- average normalized error estimate squared (ANEES)
- average normalized innovation squared (ANIS)

The project also investigates how observability changes when the measurement model and dynamics model are modified.

## Features

- Linear Kalman Filter consistency analysis
- 49-case covariance scaling study over process and measurement noise
- Heatmap visualization of MSE, ANEES, and ANIS
- Observability matrix construction and rank analysis
- Singular value decomposition of observability structure
- Comparison of original and modified measurement/dynamics models

## How to Run

Generate the truth data:

```bash
python src/util.py
```

Run the estimator and analysis:

```bash
python src/kalman_filter.py
```

## Report

See full writeup and results in:

```report/filter_analysis.pdf```