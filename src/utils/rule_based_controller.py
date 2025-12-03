import numpy as np


class RuleBasedController:
    """
    Percentile-based + SoC-aware rule controller for a BESS.

    Improvements over the simple linear controller:
    - Uses robust percentile thresholds (p_low / p_high) instead of global min/max.
    - Avoids charging when SoC is high (>= soc_max).
    - Avoids discharging when SoC is low (<= soc_min).
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
        use_observed_price: bool = True,
        soc_min: float = 0.10,
        soc_max: float = 0.90,
        q_low: float = 20.0,
        q_high: float = 80.0,
        deadband: float = 0.05,
    ):
        """
        Parameters
        ----------
        env : BatteryEnv
            Environment instance.
            
        use_observed_price : bool
            If True -> use noisy observed price from the observation.
            
        soc_min : float
            Controller-level minimum SoC threshold below which discharging is forbidden.
            (Usually chosen slightly above env.soc_min as a safety margin.)
            
        soc_max : float
            Controller-level maximum SoC threshold above which charging is forbidden.
            (Usually chosen slightly below env.soc_max as a safety margin.)
            
        q_low : float
            Lower percentile (e.g. 20th) for defining "cheap" prices.
            
        q_high : float
            Upper percentile (e.g. 80th) for defining "expensive" prices.
            
        deadband : float
            If |factor| < deadband, the controller outputs zero power.
        """
        self.env = env
        self.use_observed_price = use_observed_price
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.deadband = deadband

        # Convert price series to array
        prices = np.asarray(env.price_series, dtype=float)

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
        else:
            return float(self.env.price_series[self.env.t])

    # ------------------------------------------------------------------ #
    def _price_to_factor(self, price: float) -> float:
        """
        Convert raw price to a factor in [+1, -1] using percentile scaling.
        """
        # Normalize to [0, 1] based on percentile thresholds
        price_norm = (price - self.p_low) / (self.p_high - self.p_low)
        price_norm = float(np.clip(price_norm, 0.0, 1.0))

        # Map to factor: cheap -> +1, expensive -> -1
        return 1.0 - 2.0 * price_norm

    # ------------------------------------------------------------------ #
    def _would_violate_soc(self, soc: float, p_cmd: float) -> bool:
        """
        Predict whether applying p_cmd [kW] for one step would violate
        the environment's SoC bounds (env.soc_min / env.soc_max).

        Uses the same SoC update logic as the environment:
            - dt (hours)
            - capacity (kWh)
            - eta_c / eta_d
        """
        dt = self.env.dt
        capacity = self.env.capacity
        eta_c = self.env.eta_c
        eta_d = self.env.eta_d
        soc_min_env = self.env.soc_min
        soc_max_env = self.env.soc_max

        # Energy in kWh for this step
        energy_kWh = p_cmd * dt  # positive: charging, negative: discharging

        if p_cmd >= 0.0:
            delta_soc = (energy_kWh * eta_c) / capacity
        else:
            delta_soc = (energy_kWh / eta_d) / capacity

        soc_pre = soc + delta_soc
        return (soc_pre < soc_min_env) or (soc_pre > soc_max_env)

    # ------------------------------------------------------------------ #
    def act(self, obs):
        """
        Compute the control action for the current observation.

        Returns
        -------
        For continuous env:  np.array([P_kW], dtype=np.float32)
        For discrete env:    integer index (0..N-1)
        """
        soc = float(obs[0])  # SoC is at index 0
        price = self._get_current_price(obs)

        # 1) Price-based factor
        factor = self._price_to_factor(price)

        # 2) SoC safety override (controller-level band)
        if soc >= self.soc_max and factor > 0.0:
            factor = 0.0
        if soc <= self.soc_min and factor < 0.0:
            factor = 0.0

        # 3) Deadband (avoid small pointless actions)
        if abs(factor) < self.deadband:
            factor = 0.0

        # 4) Clip factor
        factor = float(np.clip(factor, -1.0, 1.0))

        # 5) Convert factor into actual kW command
        p_cmd = factor * self.env.p_max

        # 6) Lookahead: avoid actions that would cause SoC constraint violations in the env
        if self._would_violate_soc(soc, p_cmd):
            # Instead of causing a violation and getting penalized, stay idle this step.
            p_cmd = 0.0

        # 7) Map to discrete or continuous action space
        if self.env.use_discrete:
            disc_values = self.env.discrete_action_values
            idx = int(np.argmin(np.abs(disc_values - p_cmd)))
            return idx
        else:
            return np.array([p_cmd], dtype=np.float32)
