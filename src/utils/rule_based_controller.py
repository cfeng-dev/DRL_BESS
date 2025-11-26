import numpy as np


class RuleBasedController:
    """
    Price-based rule controller for the BESS:
    - low prices → charging (positive power)
    - high prices → discharging (negative power)
    - works with both continuous and discrete (21-level) action spaces

    Idea:
        1. Map current price to [0, 1] based on min/max in price_series.
        2. Map [0, 1] linearly to an action factor in [+1, -1]:
               cheap  price → factor ≈ +1  (charge at +P_max)
               medium price → factor ≈  0  (idle)
               expensive   → factor ≈ -1  (discharge at -P_max)
        3. Continuous env  → use factor * P_max directly.
           Discrete env    → snap to nearest of the 21 discrete kW levels.
    """

    def __init__(self, env, use_observed_price: bool = True):
        """
        Parameters
        ----------
        env : BatteryEnv
            Environment instance.
        use_observed_price : bool
            If True → use noisy/observed price from the observation (fair baseline vs. RL).
            If False → use true price_series[env.t] (unrealistic perfect knowledge).
        """
        self.env = env
        self.use_observed_price = use_observed_price

        # Cache global min/max prices for normalization
        prices = env.price_series
        self.price_min = float(np.min(prices))
        self.price_max = float(np.max(prices))

    def _get_current_price(self, obs):
        """Return current price estimate in raw units (EUR/MWh or EUR/kWh)."""
        if self.use_observed_price:
            # obs[6] = normalized price in [0, 1]
            price_norm = float(obs[6])
            return price_norm * (self.env._max_price + 1e-6)
        else:
            # Use true (noise-free) price from the underlying time series
            return float(self.env.price_series[self.env.t])

    def act(self, obs):
        """
        Compute an action (continuous or discrete, depending on env.use_discrete).

        Continuous env:
            returns np.array([P_kW], dtype=np.float32)

        Discrete env (21 actions by default):
            returns integer index into env.discrete_action_values
        """
        price = self._get_current_price(obs)

        # 1) Normalize price to [0, 1] based on global min/max
        price_norm = (price - self.price_min) / (self.price_max - self.price_min + 1e-6)
        price_norm = float(np.clip(price_norm, 0.0, 1.0))

        # 2) Map normalized price to factor in [+1, -1]
        #    cheap  (price_norm≈0) → factor≈+1  → charge at +P_max
        #    medium (≈0.5)        → factor≈0   → idle
        #    expensive (≈1)       → factor≈-1  → discharge at -P_max
        factor = 1.0 - 2.0 * price_norm

        # desired power command in kW
        p_cmd = factor * self.env.p_max

        # 3) Map to actual action format
        if self.env.use_discrete:
            # Discrete env: choose nearest of the 21 discrete kW levels
            disc_values = self.env.discrete_action_values  # e.g. [-10, -9, ..., 9, 10]
            idx = int(np.argmin(np.abs(disc_values - p_cmd)))
            return idx
        else:
            # Continuous env: return 1D array [P_kW]
            return np.array([p_cmd], dtype=np.float32)
