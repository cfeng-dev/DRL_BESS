#!/usr/bin/env python
"""
train_sac_continue.py

Train or continue training a Soft Actor-Critic (SAC) agent on the BatteryEnv.

Behavior:
- If no previous model exists -> start training from scratch.
- If a model exists and the environment "signature" matches -> load and continue.
- If a model exists but the signature DOES NOT match -> start from scratch
  (old model/replay buffer are ignored and overwritten).

This avoids accidentally continuing training with changed environment parameters.
"""

import argparse
import json
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.data_loader import load_price_data
from envs.bess_env import BatteryEnv


# ---------------------------------------------------------------------------
# Paths / Configuration (pathlib-based, independent of CWD)
# ---------------------------------------------------------------------------

# this file: .../DRL_BESS/src/training/train_sac_continue.py
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]        # -> .../DRL_BESS

DATA_CSV = PROJECT_ROOT / "data" / "raw" / "dayahead_2024_11.csv"
RESOLUTION = "1h"   # or "15min" if you also trained with 15-minute data

# All models & replay buffers go into: DRL_BESS/models/sac_baseline/
MODEL_DIR = PROJECT_ROOT / "models" / "sac_baseline"
MODEL_PATH = MODEL_DIR / "sac_bess_model.zip"
BUFFER_PATH = MODEL_DIR / "sac_bess_replay.pkl"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
SIGNATURE_PATH = MODEL_DIR / "sac_bess_signature.json"
META_PATH = MODEL_DIR / "sac_bess_meta.json"


# ---------------------------------------------------------------------------
# Helper: build + compare environment signature
# ---------------------------------------------------------------------------

def build_env_signature(env: BatteryEnv) -> dict:
    """
    Build a dictionary that captures the essential environment configuration.
    If this changes, we should NOT continue training from an old model.
    """
    sig = {
        "signature_version": 1,
        # core physical parameters
        "dt_hours": env.dt,
        "capacity_kWh": env.capacity,
        "p_max_kW": env.p_max,
        "eta_c": env.eta_c,
        "eta_d": env.eta_d,
        "soc_min": env.soc_min,
        "soc_max": env.soc_max,
        "soh_min": env.soh_min,
        "soh_deg_per_EFC": getattr(env, "soh_deg_per_EFC", None),
        "deg_cost_per_EFC": env.deg_cost_per_EFC,
        # price / demand modelling
        "price_sigma_rel": env.price_sigma_rel,
        "price_unit": env.price_unit,
        "use_simple_cycle_count": env.use_simple_cycle_count,
        # observation / action space
        "obs_shape": list(env.observation_space.shape),
        "act_shape": list(env.action_space.shape),
        "act_low": env.action_space.low.tolist(),
        "act_high": env.action_space.high.tolist(),
    }
    return sig

def load_stored_signature() -> dict | None:
    if not SIGNATURE_PATH.exists():
        return None
    try:
        with SIGNATURE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read signature file ({e}). Ignoring it.")
        return None


def signatures_match(current: dict, stored: dict | None) -> bool:
    if stored is None:
        return False
    # simple equality check; could be made more flexible if needed
    # we only compare keys that exist in the current signature
    for k, v in current.items():
        if k not in stored or stored[k] != v:
            print(f"[INFO] Signature mismatch in key '{k}': "
                  f"stored={stored.get(k)!r}, current={v!r}")
            return False
    return True


# ---------------------------------------------------------------------------
# Helper: load + save meta
# ---------------------------------------------------------------------------

def load_meta_timesteps() -> int:
    """Load total trained timesteps from META_PATH, if available."""
    if not META_PATH.exists():
        return 0
    try:
        with META_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("total_trained_timesteps", 0))
    except Exception as e:
        print(f"[WARN] Could not read meta file ({e}). Assuming 0 timesteps.")
        return 0


def save_meta_timesteps(total_timesteps: int) -> None:
    """Save total trained timesteps to META_PATH."""
    data = {"total_trained_timesteps": int(total_timesteps)}
    try:
        with META_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Saved meta information to {META_PATH}")
    except Exception as e:
        print(f"[WARN] Could not save meta file ({e}).")

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env() -> BatteryEnv:
    """Create a BatteryEnv instance with the SAME parameters as during training."""
    df, price_series, timestamps = load_price_data(
        DATA_CSV,
        resolution=RESOLUTION,
    )

    env = BatteryEnv(
        price_series=price_series,
        timestamps=timestamps,
        dt_hours=1.0,
        capacity_kWh=100.0,
        p_max_kW=50.0,
        eta_c=0.95,
        eta_d=0.95,
        soc_min=0.10,
        soc_max=0.90,
        soh_min=0.70,
        soh_deg_per_EFC=0.01,
        price_sigma_rel=0.05,
        price_unit="EUR_per_MWh",
        deg_cost_per_EFC=100.0,
        use_simple_cycle_count=True,
        random_seed=42,
    )
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=3_000,
        help="Number of additional training timesteps for this run.",
    )
    args = parser.parse_args()

    # Ensure all needed directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Data CSV    : {DATA_CSV}")
    print(f"[INFO] Model dir   : {MODEL_DIR}")

    env = make_env()

    # Build current environment signature
    current_sig = build_env_signature(env)
    stored_sig = load_stored_signature()

    # Load previous total timesteps (across all runs)
    prev_total_timesteps = load_meta_timesteps()

    # Decide: fresh training or continue from existing model
    can_continue = MODEL_PATH.exists() and signatures_match(current_sig, stored_sig)

    if can_continue:
        print(f"[INFO] Existing model found at {MODEL_PATH} and signature matches.")
        print("[INFO] Loading model and continuing training...")
        model = SAC.load(str(MODEL_PATH), env=env)
        reset_flag = False

        # Try to load replay buffer (optional)
        if BUFFER_PATH.exists():
            try:
                model.load_replay_buffer(str(BUFFER_PATH))
                print(f"[INFO] Loaded replay buffer from {BUFFER_PATH}.")
            except Exception as e:
                print(f"[WARN] Failed to load replay buffer ({e}). "
                      f"Continuing without previous buffer.")
        else:
            print("[INFO] No replay buffer found. Continuing with empty buffer.")

        reset_flag = False  # keep global timestep counter
    else:
        if MODEL_PATH.exists():
            print("[INFO] Existing model found, but environment signature differs.")
            print("[INFO] Starting NEW training from scratch and overwriting old model.")
            #prev_total_timesteps = 0
        else:
            print("[INFO] No existing model found. Starting training from scratch...")
            #prev_total_timesteps = 0
    
        model = SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=256,
            gamma=0.995,
            tau=0.02,
        )
        reset_flag = True  # start timestep counter from 0
        prev_total_timesteps = 0  # reset the external counter

    # -----------------------------------------------------------------------
    # Callbacks (optional)
    # -----------------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="sac_bess",
        save_replay_buffer=False,  # we handle replay buffer separately below
    )

    # -----------------------------------------------------------------------
    # Training / continuing training
    # -----------------------------------------------------------------------
    print(f"[INFO] Starting learn() for {args.timesteps} timesteps "
          f"(reset_num_timesteps={reset_flag})...")
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=False,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_flag,
    )

    # -----------------------------------------------------------------------
    # Save model, replay buffer, and updated signature
    # -----------------------------------------------------------------------
    model.save(str(MODEL_PATH))
    print(f"[INFO] Saved model to {MODEL_PATH}")

    try:
        model.save_replay_buffer(str(BUFFER_PATH))
        print(f"[INFO] Saved replay buffer to {BUFFER_PATH}")
    except Exception as e:
        print(f"[WARN] Could not save replay buffer ({e}).")

    # Save (or overwrite) environment signature
    try:
        with SIGNATURE_PATH.open("w", encoding="utf-8") as f:
            json.dump(current_sig, f, indent=2)
        print(f"[INFO] Saved environment signature to {SIGNATURE_PATH}")
    except Exception as e:
        print(f"[WARN] Could not save signature file ({e}).")

    # Update and save external total timestep counter
    total_trained = prev_total_timesteps + args.timesteps
    save_meta_timesteps(total_trained)
    
    print(f"[INFO] Timesteps in this run             : {args.timesteps:,}")
    print(f"[INFO] Total trained timesteps (all runs): {total_trained:,}")
    print("[INFO] Training run finished.")


if __name__ == "__main__":
    main()
