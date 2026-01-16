import numpy as np


class RuleBasedController:
    """
    Percentile-based + SoC-aware rule controller for a BESS.

    Improvements over the simple linear controller:
    - Uses robust percentile thresholds (p_low / p_high) instead of global min/max.
    - Avoids charging when SoC is high (>= soc_max_safe).
    - Avoids discharging when SoC is low (<= soc_min_safe).
    - Adds a deadband zone to prevent unnecessary switching.
    - Performs a SoC lookahead using the environment's dynamics to avoid SoC violations.
    - Works for both discrete and continuous action spaces.

    Behavior:
        price <= p_low           → strong charging (+P_max)
        price >= p_high          → strong discharging (-P_max)
        p_low < price < p_high   → linear interpolation
    """

    def __init__(
        self,
        env,
        price_history=None,
        use_observed_price: bool = True,
        soc_min: float = 0.0,
        soc_max: float = 1.0,
        q_low: float = 20.0,
        q_high: float = 80.0,
        deadband: float = 0.05,
        eps: float = 0.00,
        # Choose which env bounds to use for lookahead safety check
        # "hard" = env physical hard bounds (recommended for fair comparison)
        # "soft" = env comfort band bounds (discourage boundary operation)
        lookahead_bounds: str = "soft",
    ):
        """
        Parameters
        ----------
        env : BatteryEnv
            Environment instance.

        price_history : array-like | None
            Historical price series used to compute robust percentiles.
            If None, env.price_series is used (may be unfair).

        use_observed_price : bool
            If True -> use normalized price from observation (obs[6]) and scale back to EUR/MWh.

        soc_min : float
            Controller-level nominal minimum SoC threshold (for override logic).

        soc_max : float
            Controller-level nominal maximum SoC threshold (for override logic).

        q_low : float
            Lower percentile (e.g. 20th) for defining "cheap" prices.

        q_high : float
            Upper percentile (e.g. 80th) for defining "expensive" prices.

        deadband : float
            If |factor| < deadband, the controller outputs zero power.

        eps : float
            Safety margin for SoC bounds. The controller will internally use
            [soc_min + eps, soc_max - eps] to stay away from limits.

        lookahead_bounds : str
            Which environment bounds to use for the 1-step lookahead safety check:
            - "hard": env.soc_hard_min/max or env.soc_min/max (backward compatible)
            - "soft": env.soc_soft_min/max if available (else fallback to hard)
        """
        self.env = env
        self.use_observed_price = use_observed_price

        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.deadband = float(deadband)
        self.eps = float(eps)
        self.lookahead_bounds = str(lookahead_bounds).lower().strip()

        if self.lookahead_bounds not in ("hard", "soft"):
            raise ValueError("lookahead_bounds must be 'hard' or 'soft'.")

        # Internal "safe" band used by the controller (override logic)
        self.soc_min_safe = self.soc_min + self.eps
        self.soc_max_safe = self.soc_max - self.eps

        # Convert price series to array
        if price_history is None:
            print(
                "[WARNING] RuleBasedController: No price_history provided. "
                "Using current environment prices. This may give unfair evaluation results."
            )
            prices = np.asarray(env.price_series, dtype=float)
        else:
            prices = np.asarray(price_history, dtype=float)

        # Compute robust percentile thresholds
        self.p_low = float(np.percentile(prices, q_low))
        self.p_high = float(np.percentile(prices, q_high))

        # Safety: avoid division by zero if percentiles collapse
        if abs(self.p_high - self.p_low) < 1e-6:
            self.p_high = self.p_low + 1e-3

    # ------------------------------------------------------------------ #
    def _get_current_price(self, obs):
        """Return current (possibly noisy) price estimate."""
        if self.use_observed_price:
            price_norm = float(obs[6])  # normalized price in [0,1]
            return price_norm * (self.env._max_price + 1e-6)
        return float(self.env.price_series[self.env.t])

    # ------------------------------------------------------------------ #
    def _price_to_factor(self, price: float) -> float:
        """
        Convert raw price to a factor in [+1, -1] using percentile scaling.
        """
        price_norm = (price - self.p_low) / (self.p_high - self.p_low)
        price_norm = float(np.clip(price_norm, 0.0, 1.0))

        # cheap -> +1, expensive -> -1
        return 1.0 - 2.0 * price_norm

    # ------------------------------------------------------------------ #
    def _resolve_env_soc_bounds(self):
        """
        Resolve SoC bounds from env in a backward-compatible way.

        Returns
        -------
        (soc_min_env, soc_max_env) : tuple[float, float]
            Bounds used for the lookahead safety check.
        """
        # Old env API: soc_min/soc_max
        has_old = hasattr(self.env, "soc_min") and hasattr(self.env, "soc_max")

        # New env API: hard + soft bounds
        has_hard = hasattr(self.env, "soc_hard_min") and hasattr(self.env, "soc_hard_max")
        has_soft = hasattr(self.env, "soc_soft_min") and hasattr(self.env, "soc_soft_max")

        if self.lookahead_bounds == "soft" and has_soft:
            soc_min_env = float(self.env.soc_soft_min)
            soc_max_env = float(self.env.soc_soft_max)
            return soc_min_env + self.eps, soc_max_env - self.eps

        # default/fallback: hard bounds (preferred for fair comparison)
        if has_hard:
            soc_min_env = float(self.env.soc_hard_min)
            soc_max_env = float(self.env.soc_hard_max)
            return soc_min_env + self.eps, soc_max_env - self.eps

        if has_old:
            soc_min_env = float(self.env.soc_min)
            soc_max_env = float(self.env.soc_max)
            return soc_min_env + self.eps, soc_max_env - self.eps

        raise AttributeError(
            "Env has no SoC bound attributes (expected soc_hard_min/max, soc_soft_min/max, or soc_min/max)."
        )

    # ------------------------------------------------------------------ #
    def _would_violate_soc(self, soc: float, p_cmd: float) -> bool:
        """
        Predict whether applying p_cmd [kW] for one step would violate
        (or come too close to) the chosen env SoC bounds.

        Uses the same SoC update logic as the environment:
            - dt (hours)
            - capacity (kWh)
            - eta_c / eta_d
        """
        dt = float(self.env.dt)
        capacity = float(self.env.capacity)
        eta_c = float(self.env.eta_c)
        eta_d = float(self.env.eta_d)

        soc_min_env, soc_max_env = self._resolve_env_soc_bounds()

        # Energy in kWh for this step
        energy_kWh = p_cmd * dt  # positive: charging, negative: discharging

        if p_cmd >= 0.0:
            delta_soc = (energy_kWh * eta_c) / capacity
        else:
            delta_soc = (energy_kWh / eta_d) / capacity

        soc_pre = soc + delta_soc
        return (soc_pre <= soc_min_env) or (soc_pre >= soc_max_env)

    # ------------------------------------------------------------------ #
    def act(self, obs):
        """
        Compute the control action for the current observation.

        Returns
        -------
        For continuous env:  np.array([P_kW], dtype=np.float32)
        For discrete env:    integer index (0..N-1)
        """
        soc = float(obs[0])
        price = self._get_current_price(obs)

        # 1) Price-based factor
        factor = self._price_to_factor(price)

        # 2) SoC safety override (controller-level safe band)
        if soc >= self.soc_max_safe and factor > 0.0:
            factor = 0.0
        if soc <= self.soc_min_safe and factor < 0.0:
            factor = 0.0

        # 3) Deadband
        if abs(factor) < self.deadband:
            factor = 0.0

        # 4) Clip factor
        factor = float(np.clip(factor, -1.0, 1.0))

        # 5) Convert factor into kW command
        p_cmd = factor * float(self.env.p_max)

        # 6) Lookahead: avoid actions that would cause SoC violations
        if self._would_violate_soc(soc, p_cmd):
            p_cmd = 0.0

        # 7) Map to discrete or continuous action space
        if bool(self.env.use_discrete):
            disc_values = np.asarray(self.env.discrete_action_values, dtype=float)
            idx = int(np.argmin(np.abs(disc_values - p_cmd)))
            return idx

        return np.array([p_cmd], dtype=np.float32)
