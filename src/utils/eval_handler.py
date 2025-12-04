import numpy as np

def evaluate_financials(revenue_list, deg_cost_list, penalty_list, verbose=True):
    """
    Compute financial metrics for a BESS rollout.

    Parameters
    ----------
    revenue_list : list of float
        Money gained/lost by buying/selling energy at each timestep.
    deg_cost_list : list of float
        Cost caused by battery degradation at each timestep.
    penalty_list : list of float
        SoC/SoH constraint penalties.
    verbose : bool
        If True, prints a summary.

    Returns
    -------
    dict containing:
        - total_revenue
        - total_degradation
        - total_penalties
        - total_profit
    """

    total_revenue = float(sum(revenue_list))
    total_degradation = float(sum(deg_cost_list))
    total_penalties = float(sum(penalty_list))

    total_profit = total_revenue - total_degradation + total_penalties

    results = {
        "total_revenue": total_revenue,
        "total_degradation": total_degradation,
        "total_penalties": total_penalties,
        "total_profit": total_profit,
    }

    if verbose:
        print("=== Financial Summary ===")
        print(f"Total revenue    (EUR): {total_revenue:.3f}")
        print(f"Degradation cost (EUR): {total_degradation:.3f}")
        print(f"Penalty cost     (EUR): {total_penalties:.3f}")
        print(f"Net profit       (EUR): {total_profit:.3f}")
        print("=========================")

    return results

def evaluate_rollout(model, env, n_steps=None, deterministic=True):
    """
    Run one rollout of a trained agent in a given environment and collect:
        - state trajectories (SoC, SoH)
        - rewards
        - price data
        - actions
        - financial components (revenue, degradation, penalties)

    Parameters
    ----------
    model : SB3 model or custom controller
        The trained agent (e.g. SAC, TD3, DQN, or RuleBasedController).
    env : BatteryEnv
        Evaluation environment (must already be reset).
    n_steps : int or None
        If None → rollout runs until 'terminated' or 'truncated'.
        If int  → rollout runs for at most n_steps.
    deterministic : bool
        Whether to use deterministic policy (recommended for evaluation).

    Returns
    -------
    results : dict
        Dictionary containing all rollout arrays/lists.
    """

    # Reset environment
    obs, _ = env.reset()

    # Storage
    soc_list = []
    soh_list = []
    reward_list = []
    price_true_list = []
    action_list = []

    revenue_list = []
    deg_cost_list = []
    penalty_list = []
    violated_list = []

    # Determine rollout length
    max_steps = n_steps if n_steps is not None else len(env.price_series)

    for t in range(max_steps):
        # --------------------------------------------------
        # Choose action
        # --------------------------------------------------
        if hasattr(model, "predict"):
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = model.act(obs)  # e.g. for RuleBasedController

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # --------------------------------------------------
        # Record state and reward
        # --------------------------------------------------
        soc_list.append(obs[0])
        soh_list.append(obs[1])
        reward_list.append(reward)

        # --------------------------------------------------
        # Record external info
        # --------------------------------------------------
        price_true_list.append(info["price_true"])

        # Robust handling of continuous vs discrete actions
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                # 0D array: e.g. array(3) from DQN
                action_scalar = float(action)
            else:
                # 1D array: e.g. [a] from SAC/TD3
                action_scalar = float(action[0])
        else:
            # Plain int/float
            action_scalar = float(action)

        action_list.append(action_scalar)

        revenue_list.append(info["revenue_eur"])
        deg_cost_list.append(info["deg_cost_eur"])
        penalty_list.append(info["penalty_eur"])

        # SoC violation flag (if provided by env)
        violated_list.append(bool(info.get("violated", False)))

        if terminated or truncated:
            print(f"Episode finished early at step {t}")
            break

    results = {
        "soc": soc_list,
        "soh": soh_list,
        "reward": reward_list,
        "price_true": price_true_list,
        "action": action_list,
        "revenue": revenue_list,
        "deg_cost": deg_cost_list,
        "penalty": penalty_list,
        "violated": violated_list,
    }

    return results
