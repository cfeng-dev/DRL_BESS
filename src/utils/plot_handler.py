import matplotlib.pyplot as plt
import numpy as np


def plot_bess_rollout(
    soc_list,
    soh_list,
    reward_list,
    price_list=None,
    action_list=None,
    timestamps=None,
    figsize=(12, 15)
):
    """
    Generic plotting function for BESS environment rollouts.

    Parameters
    ----------
    soc_list : List[float]
        State of Charge values over time.

    soh_list : List[float]
        State of Health values over time.

    reward_list : List[float]
        Reward values from the environment.

    price_list : Optional[List[float]]
        Electricity price series (raw EUR/MWh or normalized).

    action_list : Optional[List[float]]
        Charge/discharge actions in kW (continuous or discrete).

    timestamps : Optional[List[datetime]]
        Real-world timestamps for x-axis. If None → use step index.

    figsize : tuple
        Figure size in inches (default = (12, 15)).
    """

    # Determine x-axis
    if timestamps is not None:
        x = timestamps[:len(soc_list)]
        xlabel = "Time"
    else:
        x = np.arange(len(soc_list))
        xlabel = "Step"

    # Number of subplots (price and actions are optional)
    n_rows = 3
    if price_list is not None:
        n_rows += 1
    if action_list is not None:
        n_rows += 1

    fig, axs = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    row = 0

    # ----------------------------------------------------------------------
    # SoC plot
    # ----------------------------------------------------------------------
    line = axs[row].axhline(0.9, color="red", linestyle="--", linewidth=1.5)
    axs[row].axhline(0.1, color="red", linestyle="--", linewidth=1.5)
    line.set_label("SoC bounds")

    axs[row].plot(x, soc_list, label="SoC", color="blue")

    axs[row].set_ylabel("SoC")
    axs[row].set_title("State of Charge")
    axs[row].grid(True)
    axs[row].set_yticks([0.0, 0.1, 0.5, 0.9, 1.0])

    axs[row].legend(loc="upper right")
    row += 1

    # ----------------------------------------------------------------------
    # SoH plot
    # ----------------------------------------------------------------------
    axs[row].plot(x, soh_list, label="SoH", color="orange")
    axs[row].set_ylabel("SoH")
    axs[row].set_title("State of Health")
    axs[row].grid(True)

    axs[row].legend(loc="upper right")
    row += 1

    # ----------------------------------------------------------------------
    # Reward plot
    # ----------------------------------------------------------------------
    axs[row].plot(x, reward_list, label="Reward", color="green")
    axs[row].set_ylabel("Reward")
    axs[row].set_title("Rewards")
    axs[row].grid(True)

    axs[row].legend(loc="upper right")
    row += 1

    # ----------------------------------------------------------------------
    # Price plot (optional)
    # ----------------------------------------------------------------------
    if price_list is not None:
        axs[row].plot(x, price_list, label="Price", color="purple")
        axs[row].set_ylabel("Price [EUR/MWh]")
        axs[row].set_title("Electricity Price")
        axs[row].grid(True)

        axs[row].legend(loc="upper right")
        row += 1

    # ----------------------------------------------------------------------
    # Action plot (optional)
    # ----------------------------------------------------------------------
    if action_list is not None:
        axs[row].step(x, action_list, where="mid", label="Action", color="red")
        axs[row].set_ylabel("Action [kW]")
        axs[row].set_title("Charge/Discharge Command")
        axs[row].grid(True)

        axs[row].legend(loc="upper right")
        row += 1

    # Final x-axis label
    axs[row - 1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()
