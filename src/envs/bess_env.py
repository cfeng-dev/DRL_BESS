import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from scenarios.price_scenario import PriceScenarioGenerator


class BatteryEnv(gym.Env):
    """
    Grid-connected Battery Energy Storage System (BESS) environment
    supporting both continuous and discrete action spaces.

    States (base):
        [SoC, SoH,
         sin_time_of_day, cos_time_of_day,
         sin_day_of_year, cos_day_of_year,
         price_norm, demand_norm,
         last_action_norm]

    If use_price_forecast=True, the state is extended by a vector of
    future normalized prices (perfect foresight) for a given horizon:
        [ ..., price_forecast_norm[0], ..., price_forecast_norm[H-1] ]

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
        discrete_action_values=None,          # normalized discrete actions in [-1, 1], if None → uses 21 evenly spaced values between -1 and +1
        eta_c: float = 0.95,                  # charging efficiency (0–1)
        eta_d: float = 0.95,                  # discharging efficiency (0–1)

        # Hard physical SoC limits (state always clipped to this range) ---
        soc_hard_min: float = 0.1,            # hard minimum SoC (fraction)
        soc_hard_max: float = 0.9,            # hard maximum SoC (fraction)

        # Soft comfort band (agent may go outside, but gets increasing penalty) ---
        soc_soft_min: float = 0.1,            # soft minimum SoC (fraction)
        soc_soft_max: float = 0.9,            # soft maximum SoC (fraction)

        soh_min: float = 0.3,                 # minimum allowed state of health before termination
        initial_soc: tuple = (0.40, 0.60),    # random initial SoC range at episode start
        price_unit: str = "EUR_per_MWh",      # price unit for conversion (can also be "EUR_per_kWh")
        deg_cost_per_EFC: float = 0.1,        # degradation cost per equivalent full cycle (in EUR)
        soh_deg_per_EFC: float = 0.005,       # physical SoH loss per equivalent full cycle

        # Soft SoC penalty shape (increases the further SoC leaves [soc_soft_min, soc_soft_max]) ---
        penalty_soc_soft_k: float = 5.0,      # strength
        penalty_soc_soft_power: float = 2.0,  # 2=quadratic, 3=cubic, ...

        use_price_forecast: bool = False,     # if True → include a future price window in the observation
        forecast_horizon_hours: float = 24.0, # forecast horizon in hours (e.g. 24h)
        episode_days: float = 7.0,            # logical episode length in days (e.g. 7 for one week)
        random_start: bool = True,            # if True → start each episode at a random index in the time series
        random_seed: int | None = None,       # RNG seed for reproducibility
        scenario_gen: PriceScenarioGenerator | None = None,  # optional: deterministic noisy forecast scenarios
        scenario_id: int = 0,                                # scenario index for reproducible evaluation
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

        # Logical episode configuration (week-based horizon)
        self.episode_days = float(episode_days)
        self.random_start = bool(random_start)
        # Default number of steps per logical episode; may be adjusted in reset()
        self.episode_len_steps = int(round(self.episode_days * 24.0 / self.dt))

        # Forecast configuration
        self.use_price_forecast = bool(use_price_forecast)
        self.forecast_horizon_hours = float(forecast_horizon_hours)
        if self.use_price_forecast:
            self.forecast_horizon_steps = max(
                1, int(round(self.forecast_horizon_hours / self.dt))
            )
        else:
            self.forecast_horizon_steps = 0

        # Default discrete actions (at p_max=10 → [-10, -9, ..., 0, ..., +9, +10] kW)
        if discrete_action_values is None:
            discrete_action_values = np.linspace(-1.0, 1.0, 21).tolist()  # 21 discrete actions between -p_max and +p_max

        # Convert normalized discrete actions to actual power (kW)
        self.discrete_action_values = np.array(discrete_action_values, dtype=np.float32) * self.p_max

        self.eta_c = float(eta_c)
        self.eta_d = float(eta_d)

        # --- NEW: hard + soft SoC constraints ---
        self.soc_hard_min = float(soc_hard_min)
        self.soc_hard_max = float(soc_hard_max)
        self.soc_soft_min = float(soc_soft_min)
        self.soc_soft_max = float(soc_soft_max)

        if not (0.0 <= self.soc_hard_min < self.soc_hard_max <= 1.0):
            raise ValueError("Hard SoC bounds must satisfy 0 <= soc_hard_min < soc_hard_max <= 1.")
        if not (self.soc_hard_min <= self.soc_soft_min < self.soc_soft_max <= self.soc_hard_max):
            raise ValueError("Soft SoC bounds must lie within hard bounds and satisfy soc_soft_min < soc_soft_max.")

        self.soh_min = float(soh_min)
        self.initial_soc_range = (float(initial_soc[0]), float(initial_soc[1]))
        self.price_unit = price_unit
        self.deg_cost_per_EFC = float(deg_cost_per_EFC)
        self.soh_deg_per_EFC = float(soh_deg_per_EFC)

        self.penalty_soc_soft_k = float(penalty_soc_soft_k)
        self.penalty_soc_soft_power = float(penalty_soc_soft_power)

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(random_seed)

        # Normalization values
        self._max_price = float(np.max(self.price_series))
        self._max_demand = None if self.demand_series is None else float(np.max(self.demand_series))

        # Scenario generator for noisy price forecasts
        self.scenario_gen = scenario_gen
        self.scenario_id = int(scenario_id)
        self._forecast_z = None

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
        # Base observation dimensions (9 features)
        base_low = np.array(
            [0.0, self.soh_min, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0],
            dtype=np.float32,
        )
        base_high = np.array(
            [1.0, 1.0, +1.0, +1.0, +1.0, +1.0, 1.0, 1.0, +1.0],
            dtype=np.float32,
        )

        if self.use_price_forecast and self.forecast_horizon_steps > 0:
            # Additional future price features, each normalized in [0, 1]
            extra_low = np.zeros(self.forecast_horizon_steps, dtype=np.float32)
            extra_high = np.ones(self.forecast_horizon_steps, dtype=np.float32)
            low = np.concatenate([base_low, extra_low])
            high = np.concatenate([base_high, extra_high])
        else:
            low = base_low
            high = base_high

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize internal state
        self.reset()

    # ----------------------------------------------------
    # RESET
    # ----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Choose logical episode start index (e.g. random week in the time series)
        # Adjust episode_len_steps if the time series is shorter than the desired episode length.
        max_episode_len = int(round(self.episode_days * 24.0 / self.dt))
        if self.T <= max_episode_len:
            self.episode_len_steps = self.T
            self.start_idx = 0
        else:
            self.episode_len_steps = max_episode_len
            if self.random_start:
                max_start = self.T - self.episode_len_steps
                self.start_idx = int(self.np_random.integers(0, max_start + 1))
            else:
                self.start_idx = 0

        self.t = self.start_idx
        self.steps_in_episode = 0

        self.soc = float(self.np_random.uniform(self.initial_soc_range[0], self.initial_soc_range[1]))
        self.soc = float(np.clip(self.soc, self.soc_hard_min, self.soc_hard_max))

        self.soh = 1.0
        self._efc_acc = 0.0
        self._last_soc = self.soc
        self.last_action = 0.0

        if self.use_price_forecast and self.forecast_horizon_steps > 0 and self.scenario_gen is not None:
            self._forecast_z = self.scenario_gen.generate_episode_noise(
                episode_len=self.episode_len_steps,
                episode_id=self.scenario_id,
            )
        else:
            self._forecast_z = None

        price_true = float(self.price_series[self.t])
        price_obs = price_true

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

        # Observed price (no additional Gaussian noise here; scenarios handled via PriceScenarioGenerator in forecast)
        price_obs = price_true

        # ----------------------------------------
        # 3. Battery SoC update with hard saturation [0,1]
        # ----------------------------------------
        # Commanded energy in kWh (positive: charging, negative: discharging)
        energy_cmd_kWh = a * self.dt

        # Commanded SoC delta (for "intent")
        if energy_cmd_kWh >= 0.0:
            delta_soc_cmd = (energy_cmd_kWh * self.eta_c) / self.capacity
        else:
            delta_soc_cmd = (energy_cmd_kWh / self.eta_d) / self.capacity

        soc_pre_cmd = self.soc + delta_soc_cmd  # what the agent "wanted"

        # --- Apply physical saturation for the actually executed energy wrt HARD bounds ---
        if energy_cmd_kWh >= 0.0:
            # Charging: cannot exceed soc_hard_max
            soc_headroom = self.soc_hard_max - self.soc
            energy_max_kWh = (soc_headroom * self.capacity) / max(self.eta_c, 1e-6)
            energy_eff_kWh = float(np.clip(energy_cmd_kWh, 0.0, energy_max_kWh))
        else:
            # Discharging: cannot go below soc_hard_min
            soc_above_min = self.soc - self.soc_hard_min
            energy_min_kWh = - (soc_above_min * self.capacity * self.eta_d)
            energy_eff_kWh = float(np.clip(energy_cmd_kWh, energy_min_kWh, 0.0))

        # SoC change based on *effective* energy
        if energy_eff_kWh >= 0.0:
            delta_soc = (energy_eff_kWh * self.eta_c) / self.capacity
        else:
            delta_soc = (energy_eff_kWh / self.eta_d) / self.capacity

        self.soc = float(np.clip(self.soc + delta_soc, self.soc_hard_min, self.soc_hard_max))

        # --- NEW: Increasing soft penalty outside comfort band [soc_soft_min, soc_soft_max] ---
        below = max(0.0, self.soc_soft_min - self.soc)
        above = max(0.0, self.soc - self.soc_soft_max)
        soft_violation = below + above

        penalty_soc_soft = -self.penalty_soc_soft_k * (
            (below ** self.penalty_soc_soft_power) + (above ** self.penalty_soc_soft_power)
        )

        # Optional: you can track whether the agent *commanded* going outside the comfort band
        violated_soft_cmd = (soc_pre_cmd < self.soc_soft_min) or (soc_pre_cmd > self.soc_soft_max)

        # ----------------------------------------
        # 4. Battery degradation (simple EFC model)
        # ----------------------------------------
        deg_cost_eur = 0.0

        delta_soc_actual = abs(self.soc - self._last_soc)
        efc_step = delta_soc_actual / 2.0

        self._efc_acc += efc_step
        self._last_soc = self.soc

        deg_cost_eur = efc_step * self.deg_cost_per_EFC

        # SoH update (still used for termination)
        self.soh = max(self.soh_min, self.soh - self.soh_deg_per_EFC * efc_step)

        # ----------------------------------------
        # 5. Revenue from charging/discharging (based on effective energy)
        # ----------------------------------------
        if self.price_unit == "EUR_per_MWh":
            price_per_kWh = price_true / 1000.0
        else:
            price_per_kWh = price_true

        revenue_eur = -price_per_kWh * energy_eff_kWh

        # ----------------------------------------
        # 6. Penalties
        # ----------------------------------------
        penalty = 0.0
        penalty += penalty_soc_soft

        reward = float(revenue_eur - deg_cost_eur + penalty)

        # ----------------------------------------
        # 7. Advance state
        # ----------------------------------------
        self.last_action = a
        self.t += 1
        self.steps_in_episode += 1

        # Termination due to SoH limit; truncation due to time horizon / data end
        terminated = (self.soh <= self.soh_min + 1e-12)
        truncated = False

        # Truncate if logical episode horizon is reached
        if self.steps_in_episode >= self.episode_len_steps:
            truncated = True

        # Truncate if we reach the end of the available time series
        if self.t >= self.T:
            truncated = True

        obs = self._get_obs(price_obs, demand)
        info = {
            "price_true": price_true,
            "revenue_eur": revenue_eur,
            "deg_cost_eur": deg_cost_eur,
            "penalty_eur": penalty,
            "penalty_soc_soft": float(penalty_soc_soft),
            "soft_violation": float(soft_violation),
            "violated_soft_cmd": bool(violated_soft_cmd),
            "efc_cum": self._efc_acc,
            "energy_cmd_kWh": energy_cmd_kWh,
            "energy_eff_kWh": energy_eff_kWh,
            "p_kw": a,
            "soc": float(self.soc),
            "soh": float(self.soh),
        }

        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------
    # INTERNAL FUNCTIONS
    # ----------------------------------------------------
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
        """
        Construct normalized observation vector.

        If use_price_forecast=True, the observation is extended by
        a window of future normalized prices (perfect knowledge of future prices).
        """
        sin_tod, cos_tod, sin_doy, cos_doy = self._get_time_features(self.t)

        # Current observed price, normalized
        price_norm = float(price_obs) / (self._max_price + 1e-6)

        # Normalized demand (if available)
        if self._max_demand is None:
            demand_norm = 0.0
        else:
            demand_norm = (
                0.0 if demand is None else float(demand) / (self._max_demand + 1e-6)
            )

        last_action_norm = float(self.last_action / self.p_max)

        obs_components = [
            self.soc,
            self.soh,
            sin_tod,
            cos_tod,
            sin_doy,
            cos_doy,
            price_norm,
            demand_norm,
            last_action_norm,
        ]

        # Optional: append future price window (scenario-based noisy forecasts)
        if self.use_price_forecast and self.forecast_horizon_steps > 0:
            prices_future = []
            for k in range(1, self.forecast_horizon_steps + 1):
                idx = min(self.t + k, self.T - 1)  # clamp at final index
                p_true_future = float(self.price_series[idx])

                if self._forecast_z is not None and self.scenario_gen is not None:
                    step_in_ep = max(0, min(self.steps_in_episode, self.episode_len_steps - 1))
                    z = float(self._forecast_z[step_in_ep, k - 1])

                    sigma_rel = float(self.scenario_gen.sigma[k - 1])
                    sigma_abs = sigma_rel * abs(p_true_future)

                    p_used_future = p_true_future + z * sigma_abs
                else:
                    p_used_future = p_true_future

                p_norm_future = p_used_future / (self._max_price + 1e-6)
                prices_future.append(p_norm_future)

            obs_components.extend(prices_future)

        return np.array(obs_components, dtype=np.float32)

    # ----------------------------------------------------
    # RENDER
    # ----------------------------------------------------
    def render(self):
        print(
            f"t={self.t:4d}  SOC={self.soc:5.3f}  SoH={self.soh:5.3f}  "
            f"EFC_cum={self._efc_acc:6.3f}  last_a={self.last_action:6.2f} kW"
        )
