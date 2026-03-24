# RLC Circuit Simulation and State-Space Modeling

This project simulates the dynamics of an RLC circuit using a state-space formulation and forward Euler integration.

## Overview

The system is modeled using the inductor current and capacitor voltage as states. The continuous-time dynamics are numerically integrated to produce time histories of the system behavior.

The simulation includes:
- State evolution of inductor current and capacitor voltage
- Computation of resistor and capacitor currents
- Evaluation of the equivalen impedance over time

## How to Run

Run the main script:

```bash
python3 src/RLC_Circuit_Simulation.py
```

## Report

See the full writeup and results in:
```report/RLC_Circuit_Simulation.pdf```
