import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BatteryEnv(gym.Env):
    """
    Grid-connected Battery Energy Storage System (BESS) environment
    with optional price forecast uncertainty and simple degradation modeling.

    States:
        [SoC, SoH,
         sin_time_of_day, cos_time_of_day,
         sin_day_of_year, cos_day_of_year,
         price_norm, demand_norm,
         last_action_norm]

        SoC                → State of Charge (0–1)
        SoH                → State of Health (0–1)
        sin_time_of_day    → Sine of normalized time-of-day (captures daily cycles)
        cos_time_of_day    → Cosine of normalized time-of-day
        sin_day_of_year    → Sine of normalized day-of-year (captures seasonal cycles)
        cos_day_of_year    → Cosine of normalized day-of-year
        price_norm         → Normalized electricity price
        demand_norm        → Normalized demand (0 if no demand data provided)
        last_action_norm   → Last action normalized to [-1, 1]

    Actions:
        a in [-P_max, +P_max]  (kW)
        a > 0 → charge from grid (buy energy)
        a < 0 → discharge to grid (sell energy)

    Reward:
        revenue - degradation_cost - penalty_cost
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        price_series,
        demand_series=None,
        timestamps=None,
        *,
        dt_hours: float = 1.0,                # time step (hours) or 0.25 for 15min
        capacity_kWh: float = 100.0,          # nominal battery capacity
        p_max_kW: float = 50.0,               # max charge/discharge power [kW]
        eta_c: float = 0.95,                  # charging efficiency
        eta_d: float = 0.95,                  # discharging efficiency
        soc_min: float = 0.10,                # prevents deep discharge (10%)
        soc_max: float = 0.90,                # prevents overcharging   (90%)
        soh_min: float = 0.70,                # below this, the battery is considered end-of-life.
        initial_soc: tuple = (0.40, 0.60),    # initial SoC range. Battery starts randomly at 40–60% SoC
        price_sigma_rel: float = 0.05,        # relative price uncertainty
        price_unit: str = "EUR_per_MWh",      # also supports "EUR_per_kWh"
        deg_cost_per_EFC: float = 100.0,      # cost per equivalent full cycle [Euro/EFC]
        use_simple_cycle_count: bool = True,
        penalty_soc_violation: float = 10.0,
        penalty_soh_violation: float = 100.0,
        random_seed: int | None = None,
    ):
        super().__init__()

        # --- Price and demand data
        self.price_series = np.asarray(price_series, dtype=np.float32)
        self.demand_series = (
            None
            if demand_series is None
            else np.asarray(demand_series, dtype=np.float32)
        )
        self.T = int(self.price_series.shape[0])

        # Optional timestamps (e.g. from CSV DateTime index)
        # Expected: sequence of datetime-like objects of length T
        self.timestamps = None
        if timestamps is not None:
            if len(timestamps) != self.T:
                raise ValueError("timestamps length must match price_series length.")
            # We keep them as a simple list/array; pandas Timestamp also works
            self.timestamps = list(timestamps)

        # --- Parameters
        self.dt = float(dt_hours)
        self.capacity = float(capacity_kWh)
        self.p_max = float(p_max_kW)
        self.eta_c = float(eta_c)
        self.eta_d = float(eta_d)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.soh_min = float(soh_min)
        self.initial_soc_range = (float(initial_soc[0]), float(initial_soc[1]))
        self.price_sigma_rel = float(price_sigma_rel)
        self.price_unit = price_unit
        self.deg_cost_per_EFC = float(deg_cost_per_EFC)
        self.use_simple_cycle_count = bool(use_simple_cycle_count)
        self.penalty_soc_violation = float(penalty_soc_violation)
        self.penalty_soh_violation = float(penalty_soh_violation)

        # --- RNG
        self.np_random, _ = gym.utils.seeding.np_random(random_seed)

        # --- Normalization caches
        self._max_price = float(np.max(self.price_series))
        self._max_demand = (
            None if self.demand_series is None else float(np.max(self.demand_series))
        )

        # --- Action Space
        # Action: [ power_command ]  in kW
        self.action_space = spaces.Box(
            low=np.array([-self.p_max], dtype=np.float32),
            high=np.array([+self.p_max], dtype=np.float32),
            dtype=np.float32,
        )

        # --- Observation Space:
        # States:
        #   [ SoC, SoH,
        #     sin_time_of_day, cos_time_of_day,
        #     sin_day_of_year, cos_day_of_year,
        #     price_norm, demand_norm,
        #     last_action_norm ]
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, self.soh_min,  -1.0, -1.0,  -1.0, -1.0,  0.0, 0.0, -1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0,          1.0,  1.0,   1.0,  1.0,   1.0, 1.0,  1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # --- State variables
        self.t: int = 0
        self.soc: float = 0.5
        self.soh: float = 1.0
        self._efc_acc: float = 0.0  # accumulated Equivalent Full Cycles
        self._last_soc: float = self.soc
        self.last_action: float = 0.0  # last action in kW

        # Initialize environment
        self.reset()

    # --------------------------------------------------------------------
    # Gymnasium API
    # --------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.t = 0
        self.soc = float(
            self.np_random.uniform(self.initial_soc_range[0], self.initial_soc_range[1])
        )
        self.soh = 1.0
        self._efc_acc = 0.0
        self._last_soc = self.soc
        self.last_action = 0.0  # no previous action at episode start

        price_true = float(self.price_series[0])
        price_obs = self._noisy_price(price_true)
        obs = self._get_obs(price_obs, None)
        return obs, {}

    def step(self, action):
        # Clamp action to valid range
        a = float(np.clip(action[0], -self.p_max, self.p_max))

        # True price and demand at current time step
        price_true = float(self.price_series[self.t])
        demand = (
            None if self.demand_series is None else float(self.demand_series[self.t])
        )

        # Observed (noisy) price -> represents price forecast with uncertainty
        price_obs = self._noisy_price(price_true)

        # Energy flow in kWh (positive → charging, negative → discharging)
        energy_kWh = a * self.dt
        if a >= 0:
            delta_soc = (energy_kWh * self.eta_c) / self.capacity
        else:
            delta_soc = (energy_kWh / self.eta_d) / self.capacity

        # SoC before clipping (used to detect constraint violations)
        soc_pre = self.soc + delta_soc
        violated = (soc_pre < self.soc_min) or (soc_pre > self.soc_max)

        # Apply SoC limits
        self.soc = float(np.clip(soc_pre, self.soc_min, self.soc_max))

        # Simple cycle counting (Equivalent Full Cycles approximation)
        if self.use_simple_cycle_count:
            self._efc_acc += abs(self.soc - self._last_soc) / 2.0
            self._last_soc = self.soc

        # Degradation cost in EUR
        deg_cost_eur = 0.0
        if self.use_simple_cycle_count:
            efc_step = abs(delta_soc) / 2.0
            deg_cost_eur = efc_step * self.deg_cost_per_EFC
            # Optional physical SoH degradation (disabled with factor 0.0)
            self.soh = max(self.soh_min, self.soh - 0.0 * efc_step)

        # Revenue calculation (buy/sell energy at true price)
        if self.price_unit == "EUR_per_MWh":
            price_per_kWh = price_true / 1000.0
        elif self.price_unit == "EUR_per_kWh":
            price_per_kWh = price_true
        else:
            raise ValueError("price_unit must be 'EUR_per_MWh' or 'EUR_per_kWh'.")

        # a > 0: buying energy (negative revenue)
        # a < 0: selling energy (positive revenue)
        revenue_eur = -price_per_kWh * energy_kWh

        # Penalties for constraint violations
        penalty = 0.0
        if violated:
            penalty -= self.penalty_soc_violation
        if self.soh <= self.soh_min + 1e-12:
            penalty -= self.penalty_soh_violation

        # Total reward
        reward = float(revenue_eur - deg_cost_eur + penalty)

        # Update last_action (for next state)
        self.last_action = a

        # Advance time
        self.t += 1
        terminated = (self.t >= self.T) or (self.soh <= self.soh_min + 1e-12)
        truncated = False

        obs = self._get_obs(price_obs, demand)
        info = {
            "price_true": price_true,
            "revenue_eur": revenue_eur,
            "deg_cost_eur": deg_cost_eur,
            "penalty_eur": penalty,
            "efc_cum": self._efc_acc,
        }
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------------------
    def _noisy_price(self, price_true: float) -> float:
        """Add Gaussian noise to the true price to model forecast uncertainty."""
        sigma = self.price_sigma_rel * abs(price_true)
        return float(self.np_random.normal(price_true, sigma))

    def _get_time_features(self, t: int):
        """
        Compute cyclic time features:
        - time-of-day as sine/cosine
        - day-of-year as sine/cosine

        If real timestamps are provided, they are used.
        Otherwise, a synthetic time index based on dt and episode length is used.
        """
        if self.timestamps is not None:
            ts = self.timestamps[t]
            # Convert pandas Timestamp to Python datetime if needed
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()

            day_of_year = ts.timetuple().tm_yday          # 1..365 (ignore leap year)
            seconds_in_day = ts.hour * 3600 + ts.minute * 60 + ts.second

            phase_day = seconds_in_day / (24.0 * 3600.0)  # [0, 1)
            phase_year = (day_of_year - 1) / 365.0        # [0, 1)
        else:
            # Fallback: purely index-based cyclic encoding
            steps_per_day = max(1, int(round(24.0 / self.dt)))
            steps_per_year = max(1, int(round(365.0 * 24.0 / self.dt)))
            phase_day = (t % steps_per_day) / steps_per_day
            phase_year = (t % steps_per_year) / steps_per_year

        sin_tod = math.sin(2.0 * math.pi * phase_day)
        cos_tod = math.cos(2.0 * math.pi * phase_day)
        sin_doy = math.sin(2.0 * math.pi * phase_year)
        cos_doy = math.cos(2.0 * math.pi * phase_year)
        return sin_tod, cos_tod, sin_doy, cos_doy

    def _get_obs(self, price_obs: float, demand: float | None):
        # Time features from current step index
        sin_tod, cos_tod, sin_doy, cos_doy = self._get_time_features(self.t)

        # Normalize price and demand to [0, 1]
        price_norm = float(price_obs) / (self._max_price + 1e-6)
        if self._max_demand is None:
            demand_norm = 0.0
        else:
            demand_norm = (
                0.0 if demand is None else float(demand) / (self._max_demand + 1e-6)
            )

        # Normalize last action to [-1, 1]
        if self.p_max > 0.0:
            last_action_norm = float(self.last_action) / self.p_max
        else:
            last_action_norm = 0.0

        obs = np.array(
            [
                self.soc,
                self.soh,
                sin_tod,
                cos_tod,
                sin_doy,
                cos_doy,
                price_norm,
                demand_norm,
                last_action_norm,
            ],
            dtype=np.float32,
        )
        return obs

    def render(self):
        print(
            f"t={self.t:4d}  SOC={self.soc:5.3f}  SoH={self.soh:5.3f}  "
            f"EFC_cum={self._efc_acc:6.3f}  last_a={self.last_action:6.2f} kW"
        )
