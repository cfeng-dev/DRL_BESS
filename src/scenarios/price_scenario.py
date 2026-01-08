# scenarios/price_scenario.py
import numpy as np

class PriceScenarioGenerator:
    """
    Generates deterministic, horizon-dependent Gaussian noise
    for electricity price forecasts.

    Noise is:
    - zero-mean Gaussian
    - sigma increases with forecast horizon
    - reproducible via base_seed
    """

    def __init__(
        self,
        horizon_steps: int,
        sigma0: float,
        sigmaH: float,
        schedule: str = "sqrt",
        base_seed: int = 42,
    ):
        self.H = horizon_steps
        self.sigma0 = float(sigma0)
        self.sigmaH = float(sigmaH)
        self.schedule = schedule
        self.base_seed = int(base_seed)

        h = np.arange(1, self.H + 1, dtype=np.float32)
        x = h / self.H

        if schedule == "linear":
            self.sigma = self.sigma0 + (self.sigmaH - self.sigma0) * x
        elif schedule == "sqrt":
            self.sigma = self.sigma0 + (self.sigmaH - self.sigma0) * np.sqrt(x)
        else:
            raise ValueError(f"Unknown sigma schedule: {schedule}")

        # Safety: avoid negative sigma due to mis-config
        self.sigma = np.maximum(self.sigma, 0.0).astype(np.float32)
        
    def generate_episode_noise(self, episode_len: int, episode_id: int) -> np.ndarray:
        """
        Returns standard normal eps with shape (episode_len, horizon_steps)
        Horizon-dependent sigma is applied in the environment
        """
        rng = np.random.default_rng(self.base_seed + episode_id)

        eps = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(episode_len, self.H),
        ).astype(np.float32)

        return eps
