# Bayesian State Estimation Projects

This repository is a collection of projects focused on state estimation, sensor fusion, and uncertainty modeling in dynamic systems. The work progresses from foundational Bayesian estimation methods to advanced nonlinear filtering applied to realistic physical systems.

The central theme across all projects is:

> How can we optimally estimate the state of a system when measurements are noisy, incomplete, and nonlinear?

## Overview

These projects cover a range of estimation techniques, including:

- Linear Kalman Filters
- Extended Kalman Filters (EKF)
- Sigma-Point Kalman Filters (UKF / SR-UKF)
- Particle Filters (PF)

Each project builds on the previous ones, introducing increasing levels of:
- system complexity
- nonlinearity
- uncertainty

The later projects emphasize full system modeling and comparative analysis between estimation methods.

---

## Project Structure

### Foundations

**1. RLC Circuit Estimation**
- Linear system modeling
- Kalman filtering fundamentals
- State and covariance propagation

**2. Discrete Bayes Filter**
- Histogram-based state estimation
- Recursive Bayesian updates
- Measurement and motion models in discrete spaces

---

### Intermediate Estimation

**3. Continuous Bayes Filter**
- Transition to continuous state spaces
- Gaussian belief representation
- Foundation for Kalman filtering

**4. Particle Filter**
- Sampling-based estimation
- Importance weighting and resampling
- Trade-offs between accuracy and computation

---

### Linear Systems (Kalman Filtering)

**5. Linear Kalman Filter I**
- System discretization (Van Loan method)
- Full KF implementation (prediction + update)
- Error analysis and covariance validation

**6. Linear Kalman Filter II**
- Consistency metrics (MSE, ANEES, ANIS)
- Sensitivity to model mismatch (Q/R tuning)
- Observability analysis using SVD

---

### Nonlinear Estimation

**7. EKF Orbit Determination**
- Nonlinear orbital dynamics
- Analytical Jacobians
- Angle wrapping and stability considerations

**8. Sigma-Point Kalman Filter (SPKF) Flight**
- Square-root UKF implementation
- Improved numerical stability
- Comparison of sigma-point methods

---

### Advanced System (Flagship Project)

**9. Vision-Based Pendulum State Estimation**
- 9-dimensional nonlinear system
- Indirect camera-based measurements
- Full system modeling (dynamics + measurement geometry)
- Implementation and comparison of:
  - EKF
  - UKF (sigma-point methods)
  - Particle Filter

This project represents a complete estimation pipeline:
- physical modeling
- nonlinear measurement inversion
- multi-method estimation
- performance comparison

---

## Key Takeaways

- Accurate state estimation requires both **model fidelity** and **uncertainty modeling**
- Kalman filter-based methods are highly efficient for systems with near-Gaussian uncertainty
- Sigma-point methods improve robustness in nonlinear systems without requiring Jacobians
- Particle filters provide flexibility at the cost of computational efficiency
- System observability and model assumptions fundamentally determine estimator performance

---

## Technologies

- Python (NumPy, SciPy, Matplotlib)
- Simulation and numerical methods
- Linear algebra and probability theory

---

## About

This repository reflects a focused effort to develop a deep understanding of state estimation in dynamic systems, with applications in areas such as robotics, aerospace, and navigation.

The emphasis is on combining:
- mathematical rigor
- physical modeling
- practical implementation

to solve real-world estimation problems under uncertainty.
