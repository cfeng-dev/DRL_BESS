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
