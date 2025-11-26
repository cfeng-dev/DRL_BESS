import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BatteryEnv(gym.Env):
    """
    Grid-connected Battery Energy Storage System (BESS) environment
    supporting both continuous and discrete action spaces.

    States:
        [SoC, SoH,
         sin_time_of_day, cos_time_of_day,
         sin_day_of_year, cos_day_of_year,
         price_norm, demand_norm,
         last_action_norm]

    Actions:
        Continuous mode:
            a ∈ [-P_max, +P_max] in kW

        Discrete mode:
            a ∈ {discrete_action_values[i] * P_max} in kW

        a > 0 → charging from grid
        a < 0 → discharging to grid
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        price_series,                         # array-like: raw electricity price time series
        demand_series=None,                   # optional: demand profile (same length as price_series)
        timestamps=None,                      # optional list of datetime objects for time features
        *,
        dt_hours: float = 0.25,               # simulation step length in hours (e.g. 1.0 for 1 hour or 0.25 for 15 min)
        capacity_kWh: float = 50.0,           # battery energy capacity in kWh (0.2C, ~5% SoC change per 15 min)
        p_max_kW: float = 10.0,               # max charge/discharge power in kW
        use_discrete_actions: bool = False,   # True → discrete action space (DQN); False → continuous (SAC/TD3)
        discrete_action_values=None,          # normalized discrete actions in [-1, 1], if None → uses 21 evenly spaced values between -1 and +1,
        eta_c: float = 0.95,                  # charging efficiency (0–1)
        eta_d: float = 0.95,                  # discharging efficiency (0–1)
        soc_min: float = 0.10,                # minimum allowed state of charge (fraction)
        soc_max: float = 0.90,                # maximum allowed state of charge (fraction)
        soh_min: float = 0.30,                # minimum allowed state of health before termination
        initial_soc: tuple = (0.40, 0.60),    # random initial SoC range at episode start
        price_sigma_rel: float = 0.05,        # price noise level (models forecast uncertainty)
        price_unit: str = "EUR_per_MWh",      # price unit for conversion (can also be "EUR_per_kWh")
        deg_cost_per_EFC: float = 1.0,        # degradation cost per equivalent full cycle (in EUR)
        soh_deg_per_EFC: float = 0.005,       # physical SoH loss per equivalent full cycle
        use_simple_cycle_count: bool = True,  # if True → simple EFC-based degradation model is applied
        penalty_soc_violation: float = 10.0,  # penalty if SoC goes outside limits (soft constraint)
        penalty_soh_violation: float = 100.0, # penalty if SoH drops below soh_min
        random_seed: int | None = None,       # RNG seed for reproducibility
    ):
        super().__init__()

        # ----------------------------------------
        # Store data
        # ----------------------------------------
        self.price_series = np.asarray(price_series, dtype=np.float32)
        self.demand_series = (
            None if demand_series is None else np.asarray(demand_series, dtype=np.float32)
        )
        self.T = len(self.price_series)

        # Optional timestamps for extracting daily/seasonal patterns
        self.timestamps = None
        if timestamps is not None:
            if len(timestamps) != self.T:
                raise ValueError("Timestamps length must match price_series length.")
            self.timestamps = list(timestamps)

        # ----------------------------------------
        # Store parameters
        # ----------------------------------------
        self.dt = float(dt_hours)
        self.capacity = float(capacity_kWh)
        self.p_max = float(p_max_kW)
        self.use_discrete = bool(use_discrete_actions)

        # Default discrete actions (at p_max=10 → [-10, -9, ..., 0, ..., +9, +10] kW)
        if discrete_action_values is None:
            discrete_action_values = np.linspace(-1.0, 1.0, 21).tolist() # 21 discrete actions between -p_max and +p_max

        # Convert normalized discrete actions to actual power (kW)
        self.discrete_action_values = np.array(discrete_action_values, dtype=np.float32) * self.p_max

        self.eta_c = float(eta_c)
        self.eta_d = float(eta_d)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.soh_min = float(soh_min)
        self.initial_soc_range = (float(initial_soc[0]), float(initial_soc[1]))
        self.price_sigma_rel = float(price_sigma_rel)
        self.price_unit = price_unit
        self.deg_cost_per_EFC = float(deg_cost_per_EFC)
        self.soh_deg_per_EFC = float(soh_deg_per_EFC)
        self.use_simple_cycle_count = bool(use_simple_cycle_count)
        self.penalty_soc_violation = float(penalty_soc_violation)
        self.penalty_soh_violation = float(penalty_soh_violation)

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(random_seed)

        # Normalization values
        self._max_price = float(np.max(self.price_series))
        self._max_demand = None if self.demand_series is None else float(np.max(self.demand_series))

        # ----------------------------------------
        # Action space: continuous OR discrete
        # ----------------------------------------
        if self.use_discrete:
            self.action_space = spaces.Discrete(len(self.discrete_action_values))
        else:
            self.action_space = spaces.Box(
                low=np.array([-self.p_max], dtype=np.float32),
                high=np.array([+self.p_max], dtype=np.float32),
                dtype=np.float32,
            )

        # ----------------------------------------
        # Observation space
        # ----------------------------------------
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, self.soh_min, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, +1.0, +1.0, +1.0, +1.0, 1.0, 1.0, +1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Initialize internal state
        self.reset()

    # ----------------------------------------------------
    # RESET
    # ----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.t = 0
        self.soc = float(self.np_random.uniform(self.initial_soc_range[0], self.initial_soc_range[1]))
        self.soh = 1.0
        self._efc_acc = 0.0
        self._last_soc = self.soc
        self.last_action = 0.0

        price_true = float(self.price_series[0])
        price_obs = self._noisy_price(price_true)

        obs = self._get_obs(price_obs, None)
        return obs, {}

    # ----------------------------------------------------
    # STEP
    # ----------------------------------------------------
    def step(self, action):
        # ----------------------------------------
        # 1. Interpret action (discrete or continuous)
        # ----------------------------------------
        if self.use_discrete:
            # Discrete action: integer index → map to actual kW command
            a = float(self.discrete_action_values[action])
        else:
            # Continuous action: take the float value directly
            a = float(np.clip(action[0], -self.p_max, self.p_max))

        # ----------------------------------------
        # 2. Fetch true values
        # ----------------------------------------
        price_true = float(self.price_series[self.t])
        demand = None if self.demand_series is None else float(self.demand_series[self.t])

        # Observed price with Gaussian noise
        price_obs = self._noisy_price(price_true)

        # ----------------------------------------
        # 3. Battery SoC update
        # ----------------------------------------
        energy_kWh = a * self.dt  # positive: charging, negative: discharging

        # Charging/discharging efficiency
        if a >= 0:
            delta_soc = (energy_kWh * self.eta_c) / self.capacity
        else:
            delta_soc = (energy_kWh / self.eta_d) / self.capacity

        # Pre-clipped SoC for penalty calculation
        soc_pre = self.soc + delta_soc
        violated = (soc_pre < self.soc_min) or (soc_pre > self.soc_max)

        # Apply SoC safety limits
        self.soc = float(np.clip(soc_pre, self.soc_min, self.soc_max))

        # ----------------------------------------
        # 4. Battery degradation (simple EFC model)
        # ----------------------------------------
        deg_cost_eur = 0.0
        if self.use_simple_cycle_count:
            delta_soc_actual = abs(self.soc - self._last_soc)
            efc_step = delta_soc_actual / 2.0

            self._efc_acc += efc_step
            self._last_soc = self.soc

            deg_cost_eur = efc_step * self.deg_cost_per_EFC

            # SoH update
            self.soh = max(self.soh_min, self.soh - self.soh_deg_per_EFC * efc_step)

        # ----------------------------------------
        # 5. Revenue from charging/discharging
        # ----------------------------------------
        if self.price_unit == "EUR_per_MWh":
            price_per_kWh = price_true / 1000.0
        else:
            price_per_kWh = price_true

        revenue_eur = -price_per_kWh * energy_kWh  # charging costs, discharging earns

        # ----------------------------------------
        # 6. Penalties
        # ----------------------------------------
        penalty = 0.0
        if violated:
            penalty -= self.penalty_soc_violation
        if self.soh <= self.soh_min + 1e-12:
            penalty -= self.penalty_soh_violation

        reward = float(revenue_eur - deg_cost_eur + penalty)

        # ----------------------------------------
        # 7. Advance state
        # ----------------------------------------
        self.last_action = a
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

    # ----------------------------------------------------
    # INTERNAL FUNCTIONS
    # ----------------------------------------------------
    def _noisy_price(self, price_true: float) -> float:
        """Add Gaussian noise to model price forecast uncertainty."""
        sigma = self.price_sigma_rel * abs(price_true)
        return float(self.np_random.normal(price_true, sigma))

    def _get_time_features(self, t: int):
        """
        Compute cyclic time features (sin/cos of time-of-day & day-of-year).
        If timestamps exist, use them. Otherwise generate synthetic time cycles.
        """
        idx = max(0, min(t, self.T - 1))

        if self.timestamps is not None:
            ts = self.timestamps[idx]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()

            day_of_year = ts.timetuple().tm_yday
            seconds_in_day = ts.hour * 3600 + ts.minute * 60 + ts.second

            phase_day = seconds_in_day / 86400.0
            phase_year = (day_of_year - 1) / 365.0
        else:
            steps_per_day = max(1, int(round(24.0 / self.dt)))
            steps_per_year = max(1, int(round(365.0 * 24.0 / self.dt)))

            phase_day = (idx % steps_per_day) / steps_per_day
            phase_year = (idx % steps_per_year) / steps_per_year

        sin_tod = math.sin(2.0 * math.pi * phase_day)
        cos_tod = math.cos(2.0 * math.pi * phase_day)
        sin_doy = math.sin(2.0 * math.pi * phase_year)
        cos_doy = math.cos(2.0 * math.pi * phase_year)

        return sin_tod, cos_tod, sin_doy, cos_doy

    def _get_obs(self, price_obs: float, demand: float | None):
        """Construct normalized observation vector."""
        sin_tod, cos_tod, sin_doy, cos_doy = self._get_time_features(self.t)

        price_norm = float(price_obs) / (self._max_price + 1e-6)

        if self._max_demand is None:
            demand_norm = 0.0
        else:
            demand_norm = (
                0.0 if demand is None else float(demand) / (self._max_demand + 1e-6)
            )

        last_action_norm = float(self.last_action / self.p_max)

        return np.array(
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

    # ----------------------------------------------------
    # RENDER
    # ----------------------------------------------------
    def render(self):
        print(
            f"t={self.t:4d}  SOC={self.soc:5.3f}  SoH={self.soh:5.3f}  "
            f"EFC_cum={self._efc_acc:6.3f}  last_a={self.last_action:6.2f} kW"
        )
