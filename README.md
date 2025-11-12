# Uncertainty-Aware Reinforcement Learning for Battery Energy Storage Control

## Overview

This repository contains the source code, simulations, and accompanying documentation for my Master’s project.  
The project investigates the use of **uncertainty-aware Deep Reinforcement Learning (DRL)** for the optimal control of a
grid-connected **Battery Energy Storage System (BESS)** under electricity price and demand uncertainty.

## Problem Description

With the increasing integration of renewable energy sources, modern power systems experience high fluctuations in
electricity prices and demand.  
BESS play a crucial role in stabilizing such systems by storing excess energy and
releasing it when needed.  
However, traditional rule-based or deterministic optimization strategies fail to account for uncertainty in price
forecasts and load predictions, leading to suboptimal charging and discharging decisions, increased battery degradation,
and reduced economic efficiency.

## Objectives

The objective of this project is to develop a DRL controller that:
- optimally operates a BESS under uncertain market and demand conditions,  
- learns continuous charge and discharge actions,  
- maximizes economic profit while minimizing battery degradation,  
- and explicitly models and incorporates forecast uncertainty in decision-making.  

The resulting framework aims to provide a robust and practical control strategy suitable for real-world applications.

## Methodology

1. **Simulation Environment**  
   Development of a realistic environment modeling a grid-connected BESS with state variables such as  
   State of Charge (SOC), State of Health (SoH), last action, forecasted prices and demand (including uncertainty),
   time of day, and day of year.

2. **Battery Model**  
   Implementation of a degradation model (e.g. Rainflow cycle counting) to quantify the effect of charge/discharge
   cycles on the SoH and convert degradation into monetary cost.

3. **RL Agents**  
   - **Baseline:** Soft Actor-Critic (SAC) for continuous action control.  
   - **Extended methods:** Uncertainty-aware approaches such as Distributional RL (QR-DQN / C51) or CVaR-optimized SAC.  
   - **Comparative models:** Rule-based control for benchmarking.

4. **Training and Evaluation**  
   - Simulations over multiple episodes (e.g. one week per episode with hourly or 15-min resolution).  
   - Performance metrics include total profit, equivalent full cycles, SoH loss, constraint violations,
     and robustness under market volatility.

## Tools

- **IDE**: JupyterLab (4.4.7)
- **Programming Language**: Python (3.13.9)
- **Package Manager**: Conda (25.5.1)
- **Libraries**:
    - pandas (2.3.3)
    - numpy (2.3.1)
    - matplotlib (3.10.6)
    - gymnasium (0.28.1)
    - stable-baselines3 (2.3.2)
    - pytorch (2.6.0)
    - pandapower (2.14.7)
    - pypsa (0.25.1)
    - scikit-learn (1.7.1)
    - scipy (1.16.0)
