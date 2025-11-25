import matplotlib.pyplot as plt
import numpy as np


def plot_bess_rollout(
    soc_list,
    soh_list,
    reward_list,
    price_list=None,
    action_list=None,
    timestamps=None,
    figsize=(12, 10)
):
    """
    Generic plotting function for BESS environment rollouts.

    Parameters:
        soc_list: List[float]
        soh_list: List[float]
        reward_list: List[float]
        price_list: Optional[List[float]]
        action_list: Optional[List[float]]  # continuous or discrete kW values
        timestamps: Optional[List[datetime]]  # if None → use step index
        figsize: tuple for figure size
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

    # SoC plot
    axs[row].plot(x, soc_list, label="SoC")
    axs[row].set_ylabel("SoC")
    axs[row].set_title("State of Charge")
    axs[row].grid(True)
    row += 1

    # SoH plot
    axs[row].plot(x, soh_list, label="SoH", color="orange")
    axs[row].set_ylabel("SoH")
    axs[row].set_title("State of Health")
    axs[row].grid(True)
    row += 1

    # Reward plot
    axs[row].plot(x, reward_list, label="Reward", color="green")
    axs[row].set_ylabel("Reward")
    axs[row].set_title("Rewards")
    axs[row].grid(True)
    row += 1

    # Price plot (optional)
    if price_list is not None:
        axs[row].plot(x, price_list, label="Price", color="purple")
        axs[row].set_ylabel("Price")
        axs[row].set_title("Electricity Price")
        axs[row].grid(True)
        row += 1

    # Action plot (optional)
    if action_list is not None:
        axs[row].step(x, action_list, where="mid", label="Action (kW)", color="red")
        axs[row].set_ylabel("Action (kW)")
        axs[row].set_title("Charge/Discharge Command")
        axs[row].grid(True)
        row += 1

    axs[row - 1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()
