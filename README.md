# Reinforcement Learning for Controlling a BESS under Uncertainty

## Overview

This repository contains the source code, simulations, and accompanying documentation for my Master’s project.  
The project investigates the use of **uncertainty-aware Deep Reinforcement Learning (DRL)** for the optimal control of a
grid-connected **Battery Energy Storage System (BESS)** under electricity price and demand uncertainty.

## Problem Description

Highly volatile electricity prices and uncertain market developments limit the efficiency of rule-based control approaches for BESS operation in the context of energy arbitrage and peak shaving. In particular, uncertainty in future load and demand patterns makes reliable planning of charging and discharging strategies for peak shaving challenging.

## Objectives

The primary objective of this Master’s project is to develop a reinforcement learning (RL) environment in Python for the control of a grid-connected BESS.

The project aims to evaluate multiple RL-based controllers to optimize BESS operation under uncertain electricity prices and fluctuating demand profiles. In particular, the behavior of different RL agents is analyzed in two operational scenarios:

- Energy arbitrage  
- Peak shaving  

## Project Components

### 1. Simulation Environment

Development of a custom Gymnasium-based environment modeling a grid-connected BESS.

The environment includes:

- State of Charge (SoC)
- State of Health (SoH)
- Current electricity price
- Current electricity demand (for peak shaving scenario)
- Time encoding (sin/cos of time-of-day and day-of-year)
- Optional price and demand forecasts
- Last executed action

### 2. Battery Model

The battery model includes:

- Power limits  
- SoC constraints  
- Charging and discharging efficiency  
- Equivalent Full Cycles (EFC) for degradation approximation

### 3. RL Agents

The following RL algorithms are compared:

- **DQN** (Discrete action space)
- **TD3** (Continuous action space)
- **QR-DQN** (Distributional RL)

Additionally:

- **Rule-Based Controller** as baseline benchmark

The agents are evaluated in both arbitrage and peak shaving scenarios.

### 4. Training and Evaluation

**Training Setup**

- Episode length: one week
- Time resolution: 15-minute intervals
- Historical electricity price and load data
- Separate train/test split  
  (November 2024 for training, first week of 2025 for testing)

**Data Sources**

- Electricity prices: [Fraunhofer ISE EnergyCharts](https://energy-charts.info)
- Load profiles: [SMARD Data Portal](https://www.smard.de/home)

**Evaluation Metrics**

- Total profit (arbitrage scenario)
- Peak reduction (peak shaving scenario)
- Final SoH

## Tools

- **IDE**: JupyterLab (4.4.7)
- **Programming Language**: Python (3.11.14)
- **Package Manager**: Conda (25.5.1)
- **Libraries**:
    - pandas (2.3.3)
    - numpy (2.3.1)
    - matplotlib (3.10.6)
    - gymnasium (0.28.1)
    - stable-baselines3 (2.3.2)
    - sb3-contrib (2.7.1)
    - pytorch (2.6.0)
    - pandapower (2.14.7)
    - protobuf (3.20.3)
    - scikit-learn (1.7.1)
    - scipy (1.16.0)
    - tensorboard (2.19.0)
    - tqdm (4.67.1)
    - rich (14.2.0)
    - ipywidgets (8.1.7)

## Required step

Run the following in the project root:

```bash
pip install -e .
```

Otherwise, imports from `src/` will not work.