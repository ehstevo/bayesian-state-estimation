# 2D Bayesian State Filtering with Multimodal Dynamics

This project implements a Bayesian filter for estimating the position of a vehicle in a two-dimensional continuous space using a discretized grid representation.

## Overview

The vehicle moves according to a stochastic, multimodal motion model, where each step consists of one of several possibilities (left, straight, or right), each corrupted by Gaussian noise. Position measurements are also noisy and modeled as Gaussian distributions.

The state is represented as a probability density function over a 2D grid. Bayesian filtering is performed recursively using:
- measurement updates via likelihood evaluation
- time propagation via 2D convolution with the process noise distribution

## Features
- Continuous-state estimation using grid-based discretization
- 2D probability density representation over a mesh
- Multimodal motion model with weighted Gaussian components
- Convolution-based propagation of uncertainty
- Separation of prior and posterior distributions
- Visualization of distribution evolutions over time

## How to Run

Run the main script:

```bash
python src/continuous_car.py
```

## Report

See the full writeup and results in:
```report/2d_continuous_bayes.pdf```