import matplotlib.pyplot as plt
import numpy as np

def plot_bess_rollout(
    soc_list,
    soh_list=None,
    reward_list=None,
    price_list=None,
    demand_list=None,
    action_list=None,
    timestamps=None,
    violated_list=None,
    figsize=(14, 24),
    fontsize_base=14, # Global font size control

    grid_import_load_kWh_list=None,   # office load supplied by grid
    supplied_to_load_kWh_list=None,   # office load supplied by BESS
    dt_hours: float = 0.25,
):
    """
    Generic plotting function for BESS environment rollouts.

    Parameters
    ----------
    soc_list : List[float]
        State of Charge values over time.
    soh_list : Optional[List[float]]
        State of Health values over time.
    reward_list : Optional[List[float]]
        Reward values from the environment.
    price_list : Optional[List[float]]
        Electricity price series (raw EUR/MWh or normalized).
    demand_list : Optional[List[float]]
        Electricity demand series (e.g. MW or MWh).
    action_list : Optional[List[float]]
        Charge/discharge actions in kW (continuous or discrete).
    timestamps : Optional[List[datetime]]
        Real-world timestamps for x-axis. If None → use step index.
    violated_list : Optional[List[bool]]
        True where the agent attempted an illegal SoC transition.
    figsize : tuple
        Figure size in inches.
    fontsize_base : int
        Base font size for all labels and titles.
    """

    # -----------------------------
    # Determine x-axis
    # -----------------------------
    n = len(soc_list)
    if timestamps is not None:
        x = timestamps[:n]
        xlabel = "Time"
    else:
        x = np.arange(n)
        xlabel = "Step"

    # -----------------------------
    # Precompute shared scaling for Demand + Peak shaving plot
    #   - shared y-axis
    #   - automatic unit switch (MWh ↔ kWh) when values are small
    # -----------------------------
    has_demand = demand_list is not None
    has_flows = (grid_import_load_kWh_list is not None) or (supplied_to_load_kWh_list is not None)

    demand_MWh = None
    e_grid_MWh = None
    e_bess_MWh = None

    if has_demand:
        demand_MWh = np.clip(np.asarray(demand_list[:n], dtype=np.float32), 0.0, None)

    if has_flows:
        if grid_import_load_kWh_list is not None:
            e_grid_MWh = np.asarray(grid_import_load_kWh_list[:n], dtype=np.float32) / 1000.0
        else:
            e_grid_MWh = np.zeros(n, dtype=np.float32)

        if supplied_to_load_kWh_list is not None:
            e_bess_MWh = np.asarray(supplied_to_load_kWh_list[:n], dtype=np.float32) / 1000.0
        else:
            e_bess_MWh = np.zeros(n, dtype=np.float32)

        e_grid_MWh = np.clip(e_grid_MWh, 0.0, None)
        e_bess_MWh = np.clip(e_bess_MWh, 0.0, None)

    # Determine common y_max in MWh (for matching y-axis)
    # If both exist -> compare both. If only one exists -> use that one.
    y_max_MWh = 0.0
    if has_demand:
        y_max_MWh = max(y_max_MWh, float(np.max(demand_MWh)))
    if has_flows:
        y_max_MWh = max(y_max_MWh, float(np.max(e_grid_MWh + e_bess_MWh)))

    # add headroom
    y_max_MWh *= 1.05 if y_max_MWh > 0 else 1.0

    # Auto unit: if small in MWh -> show kWh
    if y_max_MWh < 0.01:
        scale = 1000.0
        unit = "kWh"
    else:
        scale = 1.0
        unit = "MWh"

    y_max_scaled = y_max_MWh * scale

    # -----------------------------
    # Determine number of subplots
    # -----------------------------
    n_rows = 1  # SoC is mandatory
    if price_list is not None:
        n_rows += 1
    if demand_list is not None:
        n_rows += 1

    # Peak-shaving flow plot
    if has_flows:
        n_rows += 1

    if action_list is not None:
        n_rows += 1
    if soh_list is not None:
        n_rows += 1
    if reward_list is not None:
        n_rows += 1

    fig, axs = plt.subplots(n_rows, 1, figsize=figsize, sharex=False)
    if n_rows == 1:
        axs = [axs]

    row = 0

    # ----------------------------------------------------------------------
    # SoC plot (mandatory)
    # ----------------------------------------------------------------------
    axs[row].axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="SoC bounds")
    axs[row].axhline(0.0, color="red", linestyle="--", linewidth=1.5)

    axs[row].plot(x, soc_list, label="SoC", color="blue")

    # --- Violation markers (optional) ---
    if violated_list is not None:
        m = min(len(soc_list), len(violated_list))
        violated_indices = [i for i in range(m) if violated_list[i]]
        violated_soc = [soc_list[i] for i in violated_indices]

        axs[row].scatter(
            [x[i] for i in violated_indices],
            violated_soc,
            color="red",
            s=40,
            marker="x",
            label="Violation",
        )

    axs[row].set_ylabel("SoC", fontsize=fontsize_base)
    axs[row].set_title("State of Charge", fontsize=fontsize_base + 2, fontweight="bold")
    axs[row].grid(True)
    axs[row].set_yticks([0.0, 0.5, 1.0])
    axs[row].tick_params(axis="both", labelsize=fontsize_base)
    axs[row].legend(loc="upper right", fontsize=fontsize_base)
    row += 1

    # ----------------------------------------------------------------------
    # Price plot (optional)
    # ----------------------------------------------------------------------
    if price_list is not None:
        axs[row].plot(x, price_list[:n], label="Price", color="purple")
        axs[row].set_ylabel("Price [EUR/MWh]", fontsize=fontsize_base)
        axs[row].set_title("Electricity Price", fontsize=fontsize_base + 2, fontweight="bold")
        axs[row].grid(True)
        axs[row].tick_params(axis="both", labelsize=fontsize_base)
        axs[row].legend(loc="upper right", fontsize=fontsize_base)
        row += 1

    # ----------------------------------------------------------------------
    # Demand plot (optional)
    #   - NOW uses the SAME unit + SAME y-axis as Peak Shaving plot
    # ----------------------------------------------------------------------
    if demand_list is not None:
        demand_scaled = demand_MWh * scale
        axs[row].plot(x, demand_scaled, label="Demand", color="teal")

        axs[row].set_ylabel(f"Demand per step [{unit}]", fontsize=fontsize_base)
        axs[row].set_title("Electricity Demand", fontsize=fontsize_base + 2, fontweight="bold")
        axs[row].set_ylim(0.0, y_max_scaled)  # <<< match Peak Shaving plot
        axs[row].grid(True)
        axs[row].tick_params(axis="both", labelsize=fontsize_base)
        axs[row].legend(loc="upper right", fontsize=fontsize_base)
        row += 1

    # ----------------------------------------------------------------------
    # Grid vs BESS supply to office (Peak Shaving visualization)
    # NOTE:
    #   - Stacked area chart for clearer visualization (Grid + BESS)
    #   - Shared y-axis scaling with Demand for direct comparison
    #   - Automatic unit switch (MWh ↔ kWh) when values are very small
    # ----------------------------------------------------------------------
    if has_flows:
        e_grid = e_grid_MWh * scale
        e_bess = e_bess_MWh * scale
        total_supply = e_grid + e_bess

        axs[row].stackplot(
            x,
            e_grid,
            e_bess,
            labels=["Grid → Office", "BESS → Office"],
            alpha=0.75,
        )

        axs[row].plot(
            x,
            total_supply,
            linestyle="--",
            linewidth=2.0,
            label="Total supply",
        )

        if demand_list is not None:
            axs[row].plot(
                x,
                demand_scaled,
                linestyle=":",
                linewidth=2.5,
                label="Demand",
            )

        axs[row].set_ylim(0.0, y_max_scaled)  # <<< shared y-axis
        axs[row].set_ylabel(f"Energy per step [{unit}]", fontsize=fontsize_base)
        axs[row].set_title("Office Supply (Peak Shaving Effect)", fontsize=fontsize_base + 2, fontweight="bold")
        axs[row].grid(True)
        axs[row].tick_params(axis="both", labelsize=fontsize_base)
        axs[row].legend(loc="upper right", fontsize=fontsize_base)
        row += 1

    # ----------------------------------------------------------------------
    # Action plot (optional)
    # ----------------------------------------------------------------------
    if action_list is not None:
        axs[row].step(x, action_list[:n], where="mid", label="Action", color="red")
        axs[row].set_ylabel("Action [kW]", fontsize=fontsize_base)
        axs[row].set_title("Charge / Discharge Command", fontsize=fontsize_base + 2, fontweight="bold")
        axs[row].grid(True)
        axs[row].tick_params(axis="both", labelsize=fontsize_base)
        axs[row].legend(loc="upper right", fontsize=fontsize_base)
        row += 1

    # ----------------------------------------------------------------------
    # SoH plot (optional)
    # ----------------------------------------------------------------------
    if soh_list is not None:
        axs[row].plot(x, soh_list[:n], label="SoH", color="orange")

        last_soh = float(soh_list[min(n, len(soh_list)) - 1])
        last_x = x[min(n, len(soh_list)) - 1]

        axs[row].scatter([last_x], [last_soh], s=50, color="black", zorder=5)
        axs[row].annotate(
            f"last: {last_soh:.3f}",
            xy=(last_x, last_soh),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=fontsize_base,
        )

        axs[row].set_ylabel("SoH", fontsize=fontsize_base)
        axs[row].set_title("State of Health", fontsize=fontsize_base + 2, fontweight="bold")
        axs[row].grid(True)
        axs[row].tick_params(axis="both", labelsize=fontsize_base)
        axs[row].legend(loc="upper right", fontsize=fontsize_base)
        row += 1

    # ----------------------------------------------------------------------
    # Reward plot (optional)
    # ----------------------------------------------------------------------
    if reward_list is not None:
        axs[row].plot(x, reward_list[:n], label="Reward", color="green")
        axs[row].set_ylabel("Reward", fontsize=fontsize_base)
        axs[row].set_title("Reward", fontsize=fontsize_base + 2, fontweight="bold")
        axs[row].grid(True)
        axs[row].tick_params(axis="both", labelsize=fontsize_base)
        axs[row].legend(loc="upper right", fontsize=fontsize_base)
        row += 1

    # Add x-label to ALL subplots
    for ax in axs:
        ax.set_xlabel(xlabel, fontsize=fontsize_base)

    plt.tight_layout()
    plt.show()
