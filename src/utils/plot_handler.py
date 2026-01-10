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
    if timestamps is not None:
        x = timestamps[:len(soc_list)]
        xlabel = "Time"
    else:
        x = np.arange(len(soc_list))
        xlabel = "Step"

    # -----------------------------
    # Determine number of subplots
    # -----------------------------
    n_rows = 1  # SoC is mandatory
    if price_list is not None:
        n_rows += 1
    if demand_list is not None:
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
    axs[row].axhline(0.9, color="red", linestyle="--", linewidth=1.5, label="SoC bounds")
    axs[row].axhline(0.1, color="red", linestyle="--", linewidth=1.5)

    axs[row].plot(x, soc_list, label="SoC", color="blue")

    # --- Violation markers (optional) ---
    if violated_list is not None:
        n = min(len(soc_list), len(violated_list))
        violated_indices = [i for i in range(n) if violated_list[i]]
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
    axs[row].set_yticks([0.0, 0.1, 0.5, 0.9, 1.0])
    axs[row].tick_params(axis="both", labelsize=fontsize_base)
    axs[row].legend(loc="upper right", fontsize=fontsize_base)
    row += 1

    # ----------------------------------------------------------------------
    # Price plot (optional)
    # ----------------------------------------------------------------------
    if price_list is not None:
        axs[row].plot(x, price_list, label="Price", color="purple")
        axs[row].set_ylabel("Price [EUR/MWh]", fontsize=fontsize_base)
        axs[row].set_title("Electricity Price", fontsize=fontsize_base + 2, fontweight="bold")
        axs[row].grid(True)
        axs[row].tick_params(axis="both", labelsize=fontsize_base)
        axs[row].legend(loc="upper right", fontsize=fontsize_base)
        row += 1

    # ----------------------------------------------------------------------
    # Demand plot (optional)
    # ----------------------------------------------------------------------
    if demand_list is not None:
        axs[row].plot(x, demand_list, label="Demand", color="teal")
        axs[row].set_ylabel("Demand", fontsize=fontsize_base)
        axs[row].set_title("Electricity Demand", fontsize=fontsize_base + 2, fontweight="bold")
        axs[row].grid(True)
        axs[row].tick_params(axis="both", labelsize=fontsize_base)
        axs[row].legend(loc="upper right", fontsize=fontsize_base)
        row += 1

    # ----------------------------------------------------------------------
    # Action plot (optional)
    # ----------------------------------------------------------------------
    if action_list is not None:
        axs[row].step(x, action_list, where="mid", label="Action", color="red")
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
        axs[row].plot(x, soh_list, label="SoH", color="orange")

        # --- show last SoH value on the plot ---
        last_soh = float(soh_list[-1])
        last_x = x[len(soh_list) - 1]

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
        axs[row].plot(x, reward_list, label="Reward", color="green")
        axs[row].set_ylabel("Reward", fontsize=fontsize_base)
        axs[row].set_title("Reward", fontsize=fontsize_base + 2, fontweight="bold")
        axs[row].grid(True)
        axs[row].tick_params(axis="both", labelsize=fontsize_base)
        axs[row].legend(loc="upper right", fontsize=fontsize_base)
        row += 1

    # Final x-axis label
    axs[row - 1].set_xlabel(xlabel, fontsize=fontsize_base)
    plt.tight_layout()
    plt.show()
