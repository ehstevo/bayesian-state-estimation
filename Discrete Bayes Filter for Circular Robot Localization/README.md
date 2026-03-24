# Discrete Bayes Filter for Circular Robot Localization

This project implements a discrete Bayes filter for localization in a stochastic environment with uncertain motion and noisy measurements.

## Overview

A robot moves among 50 discrete positions arranged in a circular map, where each position is associated with a color. At each time step, the robot receives a noisy measurement of the tile color and moves according to a probabilistic motion model.

The robot's position is estimated using a recursive Bayes filter consisting of:
- measurement update using the likelihood model
- time propagation using a Markov transition model

## Features

- Discrete-state Bayesian filtering
- Probabilistic motion and measurement models
- Monte Carlo simulation for statistical evaluation
- Performance metrics:
    - accuracy (correct state estimate)
    - validity (consistency of the probability distribution)
    - goodness (confidence of the estimate)
- Sensitivity analysis under sensor model mismatch
- Parameter sweep over sensor reliability and motion models

## How to Run

Run the main script:

```bash
python src/discrete_robot.py
```
```monte_carlo()``` will run 100 iterations of estimation
uncomment the part under monte carlo to run the parameter sweep

## Report

See the full writeup and analysis in:
```report/discrete_robot.pdf```