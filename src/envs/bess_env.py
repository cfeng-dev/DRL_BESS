import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.forecast_scenario import ForecastScenarioGenerator


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
    future normalized prices for a given horizon:
        [..., price_forecast_norm[0], ..., price_forecast_norm[H-1]]

    If use_demand_forecast=True, the state is additionally extended by a vector of
    future normalized demand values for the same horizon:
        [..., demand_forecast_norm[0], ..., demand_forecast_norm[H-1]]

    If both are enabled:
        [..., price_forecast_norm[0..H-1], demand_forecast_norm[0..H-1]]

    Forecast uncertainty:
        - price forecast: relative sigma per horizon (sigma[k]) converted to absolute units via |p_true|
        - demand forecast: multiplicative relative noise

    Expose uncertainty to the agent
        - include_price_sigma=True  -> append sigma_price_norm[0..H-1]
        - include_demand_sigma=True -> append sigma_demand_norm[0..H-1]
      (sigma vectors are normalized to [0,1] by dividing by max sigma)

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
        dt_hours: float = 0.25,               # simulation step length in hours
        capacity_kWh: float = 50.0,           # battery energy capacity in kWh
        p_max_kW: float = 10.0,               # max charge/discharge power in kW
        use_discrete_actions: bool = False,   # True → discrete action space (DQN); False → continuous (SAC/TD3)
        discrete_action_values=None,          # normalized discrete actions in [-1, 1]
        eta_c: float = 0.95,                  # charging efficiency (0–1)
        eta_d: float = 0.95,                  # discharging efficiency (0–1)

        # Hard physical SoC limits
        soc_hard_min: float = 0.0,
        soc_hard_max: float = 1.0,

        # Soft comfort band (penalty outside)
        soc_soft_min: float = 0.0,
        soc_soft_max: float = 1.0,

        soh_min: float = 0.3,
        initial_soc: tuple = (0.40, 0.60),
        price_unit: str = "EUR_per_MWh",      # "EUR_per_MWh" or "EUR_per_kWh"
        deg_cost_per_EFC: float = 0.1,        # degradation cost per equivalent full cycle (EUR)
        soh_deg_per_EFC: float = 0.005,       # SoH loss per EFC

        penalty_soc_soft_k: float = 5.0,
        penalty_soc_soft_power: float = 2.0,

        # Forecast controls
        use_price_forecast: bool = False,     # Include price forecast window
        use_demand_forecast: bool = False,    # Include demand forecast window
        forecast_horizon_hours: float = 24.0,

        # Expose uncertainty to agent
        include_price_sigma: bool = False,
        include_demand_sigma: bool = False,

        episode_days: float = 7.0,
        random_start: bool = True,
        random_seed: int | None = None,

        # Scenario generators (noise)
        price_scenario_gen: ForecastScenarioGenerator | None = None,
        demand_scenario_gen: ForecastScenarioGenerator | None = None,

        scenario_id: int = 0,
        vary_scenario_per_episode: bool = True,
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

        if self.demand_series is not None and len(self.demand_series) != self.T:
            raise ValueError("demand_series length must match price_series length.")

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

        self.episode_days = float(episode_days)
        self.random_start = bool(random_start)
        self.episode_len_steps = int(round(self.episode_days * 24.0 / self.dt))

        # Forecast configuration
        self.use_price_forecast = bool(use_price_forecast)
        self.use_demand_forecast = bool(use_demand_forecast)
        self.forecast_horizon_hours = float(forecast_horizon_hours)

        # NEW: expose uncertainty flags
        self.include_price_sigma = bool(include_price_sigma)
        self.include_demand_sigma = bool(include_demand_sigma)

        # only meaningful if corresponding forecast is enabled
        if not self.use_price_forecast:
            self.include_price_sigma = False
        if not self.use_demand_forecast:
            self.include_demand_sigma = False

        if (self.use_demand_forecast and self.demand_series is None):
            raise ValueError("use_demand_forecast=True requires demand_series to be provided.")

        if self.use_price_forecast or self.use_demand_forecast:
            self.forecast_horizon_steps = max(1, int(round(self.forecast_horizon_hours / self.dt)))
        else:
            self.forecast_horizon_steps = 0

        # Discrete actions
        if discrete_action_values is None:
            discrete_action_values = np.linspace(-1.0, 1.0, 21).tolist()

        self.discrete_action_values = np.array(discrete_action_values, dtype=np.float32) * self.p_max

        self.eta_c = float(eta_c)
        self.eta_d = float(eta_d)

        # SoC constraints
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
        self._max_price = float(np.max(self.price_series)) if self.T > 0 else 1.0
        self._max_demand = None if self.demand_series is None else float(np.max(self.demand_series))

        # Scenario generators
        self.price_scenario_gen = price_scenario_gen
        self.demand_scenario_gen = demand_scenario_gen

        self.scenario_id = int(scenario_id)
        self.vary_scenario_per_episode = bool(vary_scenario_per_episode)

        self._forecast_z_price = None
        self._forecast_z_demand = None
        self._episode_counter = 0  # used to vary scenario noise across resets (training)

        # Sigma normalization (map sigma vectors into [0,1])
        self._sigma_price_max = 1.0
        self._sigma_demand_max = 1.0
        if self.price_scenario_gen is not None and hasattr(self.price_scenario_gen, "sigma"):
            sig = np.asarray(self.price_scenario_gen.sigma, dtype=np.float32)
            self._sigma_price_max = float(np.max(sig)) if sig.size > 0 else 1.0
        if self.demand_scenario_gen is not None and hasattr(self.demand_scenario_gen, "sigma"):
            sig = np.asarray(self.demand_scenario_gen.sigma, dtype=np.float32)
            self._sigma_demand_max = float(np.max(sig)) if sig.size > 0 else 1.0

        # Safety: if using forecast + scenario_gen, dimensions must match
        if self.forecast_horizon_steps > 0:
            if self.use_price_forecast and self.price_scenario_gen is not None:
                if int(self.price_scenario_gen.H) != int(self.forecast_horizon_steps):
                    raise ValueError(
                        f"price_scenario_gen.H ({self.price_scenario_gen.H}) must equal forecast_horizon_steps ({self.forecast_horizon_steps})."
                    )
            if self.use_demand_forecast and self.demand_scenario_gen is not None:
                if int(self.demand_scenario_gen.H) != int(self.forecast_horizon_steps):
                    raise ValueError(
                        f"demand_scenario_gen.H ({self.demand_scenario_gen.H}) must equal forecast_horizon_steps ({self.forecast_horizon_steps})."
                    )

        # ----------------------------------------
        # Action space
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
        base_low = np.array(
            [0.0, self.soh_min, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0],
            dtype=np.float32,
        )
        base_high = np.array(
            [1.0, 1.0, +1.0, +1.0, +1.0, +1.0, 1.0, 1.0, +1.0],
            dtype=np.float32,
        )

        extra_dim = 0
        if self.use_price_forecast and self.forecast_horizon_steps > 0:
            extra_dim += self.forecast_horizon_steps
        if self.use_demand_forecast and self.forecast_horizon_steps > 0:
            extra_dim += self.forecast_horizon_steps

        # NEW: add sigma vectors as features
        if self.include_price_sigma and self.forecast_horizon_steps > 0 and (self.price_scenario_gen is not None):
            extra_dim += self.forecast_horizon_steps
        if self.include_demand_sigma and self.forecast_horizon_steps > 0 and (self.demand_scenario_gen is not None):
            extra_dim += self.forecast_horizon_steps

        if extra_dim > 0:
            extra_low = np.zeros(extra_dim, dtype=np.float32)
            extra_high = np.ones(extra_dim, dtype=np.float32)
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

        options = {} if options is None else dict(options)

        # Episode length & start index
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

        forced_episode_id = options.get("episode_id", None)

        # Decide episode_id used by BOTH forecasts (for reproducible scenario pairing)
        if forced_episode_id is not None:
            episode_id = int(forced_episode_id)
        else:
            if self.vary_scenario_per_episode:
                episode_id = self.scenario_id + self._episode_counter
            else:
                episode_id = self.scenario_id

        # Generate noise matrices
        self._forecast_z_price = None
        self._forecast_z_demand = None

        if self.forecast_horizon_steps > 0:
            if self.use_price_forecast and self.price_scenario_gen is not None:
                self._forecast_z_price = self.price_scenario_gen.generate_episode_noise(
                    episode_len=self.episode_len_steps,
                    episode_id=episode_id,
                )
            if self.use_demand_forecast and self.demand_scenario_gen is not None:
                self._forecast_z_demand = self.demand_scenario_gen.generate_episode_noise(
                    episode_len=self.episode_len_steps,
                    episode_id=episode_id,
                )

        # increment AFTER using it
        self._episode_counter += 1

        price_true = float(self.price_series[self.t])
        price_obs = price_true

        demand = None if self.demand_series is None else float(self.demand_series[self.t])
        obs = self._get_obs(price_obs, demand)
        return obs, {}

    # ----------------------------------------------------
    # STEP
    # ----------------------------------------------------
    def step(self, action):
        # 1) Interpret action
        if self.use_discrete:
            a = float(self.discrete_action_values[action])
        else:
            a = float(np.clip(action[0], -self.p_max, self.p_max))

        # 2) Fetch true values
        price_true = float(self.price_series[self.t])
        demand_true = None if self.demand_series is None else float(self.demand_series[self.t])

        price_obs = price_true  # observed price (forecasts handled in obs)

        # 3) Battery SoC update with hard saturation
        energy_cmd_kWh = a * self.dt

        if energy_cmd_kWh >= 0.0:
            delta_soc_cmd = (energy_cmd_kWh * self.eta_c) / self.capacity
        else:
            delta_soc_cmd = (energy_cmd_kWh / self.eta_d) / self.capacity

        soc_pre_cmd = self.soc + delta_soc_cmd

        if energy_cmd_kWh >= 0.0:
            soc_headroom = self.soc_hard_max - self.soc
            energy_max_kWh = (soc_headroom * self.capacity) / max(self.eta_c, 1e-6)
            energy_eff_kWh = float(np.clip(energy_cmd_kWh, 0.0, energy_max_kWh))
        else:
            soc_above_min = self.soc - self.soc_hard_min
            energy_min_kWh = -(soc_above_min * self.capacity * self.eta_d)
            energy_eff_kWh = float(np.clip(energy_cmd_kWh, energy_min_kWh, 0.0))

        if energy_eff_kWh >= 0.0:
            delta_soc = (energy_eff_kWh * self.eta_c) / self.capacity
        else:
            delta_soc = (energy_eff_kWh / self.eta_d) / self.capacity

        self.soc = float(np.clip(self.soc + delta_soc, self.soc_hard_min, self.soc_hard_max))

        below = max(0.0, self.soc_soft_min - self.soc)
        above = max(0.0, self.soc - self.soc_soft_max)
        soft_violation = below + above

        penalty_soc_soft = -self.penalty_soc_soft_k * (
            (below ** self.penalty_soc_soft_power) + (above ** self.penalty_soc_soft_power)
        )

        violated_soft_cmd = (soc_pre_cmd < self.soc_soft_min) or (soc_pre_cmd > self.soc_soft_max)

        # 4) Degradation (simple EFC model)
        delta_soc_actual = abs(self.soc - self._last_soc)
        efc_step = delta_soc_actual / 2.0

        self._efc_acc += efc_step
        self._last_soc = self.soc

        deg_cost_eur = efc_step * self.deg_cost_per_EFC

        self.soh = max(self.soh_min, self.soh - self.soh_deg_per_EFC * efc_step)

        # 5) Revenue
        if self.price_unit == "EUR_per_MWh":
            price_per_kWh = price_true / 1000.0
        else:
            price_per_kWh = price_true

        revenue_eur = -price_per_kWh * energy_eff_kWh

        # 6) Reward
        penalty = float(penalty_soc_soft)
        reward = float(revenue_eur - deg_cost_eur + penalty)

        # 7) Advance
        self.last_action = a
        self.t += 1
        self.steps_in_episode += 1

        terminated = (self.soh <= self.soh_min + 1e-12)
        truncated = False
        if self.steps_in_episode >= self.episode_len_steps:
            truncated = True
        if self.t >= self.T:
            truncated = True

        obs = self._get_obs(price_obs, demand_true)
        info = {
            "price_true": price_true,
            "demand_true": demand_true,
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

        Forecast windows:
            - price forecast: relative sigma per horizon -> absolute noise via |p_true|
            - demand forecast: multiplicative relative noise

        NEW:
            - appends sigma vectors (normalized to [0,1]) if include_*_sigma is enabled
        """
        sin_tod, cos_tod, sin_doy, cos_doy = self._get_time_features(self.t)

        price_norm = float(price_obs) / (self._max_price + 1e-6)

        if self._max_demand is None:
            demand_norm = 0.0
        else:
            demand_norm = 0.0 if demand is None else float(demand) / (self._max_demand + 1e-6)

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

        # Noise row index for the current step inside the episode
        step_in_ep = max(0, min(self.steps_in_episode, max(0, self.episode_len_steps - 1)))

        # Append future price window (if enabled)
        if self.use_price_forecast and self.forecast_horizon_steps > 0:
            prices_future = []
            for k in range(1, self.forecast_horizon_steps + 1):
                idx = min(self.t + k, self.T - 1)
                p_true_future = float(self.price_series[idx])

                if self._forecast_z_price is not None and self.price_scenario_gen is not None:
                    eps = float(self._forecast_z_price[step_in_ep, k - 1])  # ~ N(0,1)
                    sigma_rel = float(self.price_scenario_gen.sigma[k - 1])
                    sigma_abs = sigma_rel * max(abs(p_true_future), 1e-6)  # avoid 0-scale
                    p_used_future = p_true_future + eps * sigma_abs
                else:
                    p_used_future = p_true_future

                p_norm_future = p_used_future / (self._max_price + 1e-6)
                prices_future.append(float(p_norm_future))

            obs_components.extend(prices_future)

        # Append future demand window (if enabled)
        if self.use_demand_forecast and self.forecast_horizon_steps > 0:
            demands_future = []
            for k in range(1, self.forecast_horizon_steps + 1):
                idx = min(self.t + k, self.T - 1)
                d_true_future = float(self.demand_series[idx])

                if self._forecast_z_demand is not None and self.demand_scenario_gen is not None:
                    eps = float(self._forecast_z_demand[step_in_ep, k - 1])  # ~ N(0,1)
                    sigma_rel = float(self.demand_scenario_gen.sigma[k - 1])
                    d_used_future = d_true_future * (1.0 + eps * sigma_rel)
                    d_used_future = max(d_used_future, 0.0)  # demand can't be negative
                else:
                    d_used_future = d_true_future

                d_norm_future = d_used_future / (self._max_demand + 1e-6)
                demands_future.append(float(d_norm_future))

            obs_components.extend(demands_future)

        # ----------------------------------------------------
        # Append sigma (uncertainty) vectors (if enabled)
        # ----------------------------------------------------
        if (
            self.include_price_sigma
            and self.use_price_forecast
            and self.forecast_horizon_steps > 0
            and (self.price_scenario_gen is not None)
            and hasattr(self.price_scenario_gen, "sigma")
        ):
            sig = np.asarray(self.price_scenario_gen.sigma, dtype=np.float32)
            sig_norm = sig / (self._sigma_price_max + 1e-6)
            sig_norm = np.clip(sig_norm, 0.0, 1.0)
            obs_components.extend([float(x) for x in sig_norm])

        if (
            self.include_demand_sigma
            and self.use_demand_forecast
            and self.forecast_horizon_steps > 0
            and (self.demand_scenario_gen is not None)
            and hasattr(self.demand_scenario_gen, "sigma")
        ):
            sig = np.asarray(self.demand_scenario_gen.sigma, dtype=np.float32)
            sig_norm = sig / (self._sigma_demand_max + 1e-6)
            sig_norm = np.clip(sig_norm, 0.0, 1.0)
            obs_components.extend([float(x) for x in sig_norm])

        return np.array(obs_components, dtype=np.float32)

    # ----------------------------------------------------
    # RENDER
    # ----------------------------------------------------
    def render(self):
        print(
            f"t={self.t:4d}  SOC={self.soc:5.3f}  SoH={self.soh:5.3f}  "
            f"EFC_cum={self._efc_acc:6.3f}  last_a={self.last_action:6.2f} kW"
        )
