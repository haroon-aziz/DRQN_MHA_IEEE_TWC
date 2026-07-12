
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from env.markov_chain import PhyParams
from env.wlan_env import WlanEnv, RewardWeights
from agent.drqn_agent import DRQNAgent


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_phy(cfg: dict) -> PhyParams:
    """Build PhyParams from the YAML config (Table I section)."""
    p = cfg["phy"]
    return PhyParams(
        payload_bits=p["payload_bits"],
        phy_header_bits=p["phy_header_bits"],
        mac_header_bits=p["mac_header_bits"],
        channel_bit_rate_mbps=p["channel_bit_rate_mbps"],
        slot_time_us=p["slot_time_us"],
        cw_min=p["cw_min"],
        cw_max=p["cw_max"],
        max_backoff_stages=p["max_backoff_stages"],
    )


def build_env(cfg: dict, phy: PhyParams, cache_dir: str, seed: int) -> WlanEnv:
    """Build the WlanEnv from the YAML config."""
    e = cfg["env"]
    rw = e["reward_weights"]
    weights = RewardWeights(
        alpha=rw["alpha"], beta=rw["beta"], gamma=rw["gamma"],
        delta=rw["delta"], epsilon=rw["epsilon"],
    )
    return WlanEnv(
        phy=phy,
        cache_dir=cache_dir,
        reward_weights=weights,
        action_space=tuple(e["action_space"]),
        default_ru=e["default_ru"],
        n_drift=cfg["phy"]["n_drift"],
        hol_normalize_slots=e["hol_normalize_slots"],
        ewma_alpha=e["ewma_alpha"],
        seed=seed,
    )


def build_agent(cfg: dict, device: torch.device) -> DRQNAgent:
    """Build the DRQNAgent from the YAML config."""
    m = cfg["model"]
    t = cfg["train"]
    return DRQNAgent(
        state_dim=m["state_dim"],
        n_actions=m["n_actions"],
        seq_len=t["sequence_length"],
        device=device,
        lr=t["lr"],
        gamma=t["gamma"],
        epsilon_start=t["epsilon_start"],
        epsilon_end=t["epsilon_end"],
        epsilon_decay_steps=t["epsilon_decay_steps"],
        target_update_tau=t["target_update_tau"],
        target_update_every=t["target_update_every"],
        grad_clip_norm=t["grad_clip_norm"],
        buffer_capacity=t["buffer_capacity"],
        batch_size=t["batch_size"],
        huber_delta=t["huber_loss_delta"],
    )


def train(cfg: dict, seed: int, total_steps: int | None = None,
          checkpoint_dir: str | None = None) -> str:
  
    # ----- Hardware -----
    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available — falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[INFO] Using device: {device}")

    # ----- Paths -----
    package_root = Path(__file__).resolve().parent
    cache_dir = str(package_root / ".cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (package_root / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----- Build env + agent -----
    phy = build_phy(cfg)
    env = build_env(cfg, phy, cache_dir, seed=seed)
    agent = build_agent(cfg, device)
    print(f"[INFO] Trainable parameters: "
          f"{sum(p.numel() for p in agent.online.parameters() if p.requires_grad):,}")

    # ----- Training hyperparameters -----
    t = cfg["train"]
    n_steps = total_steps if total_steps is not None else t["total_steps"]
    learning_starts = t["learning_starts"]
    train_every = t["train_every"]
    checkpoint_every = t["checkpoint_every"]
    episode_len = cfg["eval"]["episode_steps"]

    # ----- Episode management -----
    state = env.reset()
    agent.reset_state_sequence(state)
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    # Rolling metrics (last 1000 steps)
    recent_rewards: list[float] = []
    recent_S: list[float] = []
    recent_pcoll: list[float] = []

    print(f"[INFO] Starting training for {n_steps:,} steps "
          f"(seed={seed}, episode_len={episode_len})")
    print(f"[INFO] Cache dir: {cache_dir}")
    print(f"[INFO] Checkpoint dir: {ckpt_dir}")
    print("-" * 78)

    t_start = time.time()
    last_log = t_start

    for step in range(1, n_steps + 1):
        # ---- Agent acts (ε-greedy with linear decay) ----
        action = agent.observe_and_act(state, eval_mode=False, step=step - 1)
        next_state, reward, done, info = env.step(action)

        # ---- Store transition ----
        agent.store(state, action, reward, next_state, done)

        # ---- Advance ----
        state = next_state
        episode_reward += reward
        episode_steps += 1
        recent_rewards.append(reward)
        recent_S.append(info["S"])
        recent_pcoll.append(info["p_coll"])
        if len(recent_rewards) > 1000:
            recent_rewards = recent_rewards[-1000:]
            recent_S = recent_S[-1000:]
            recent_pcoll = recent_pcoll[-1000:]

        # ---- Periodic episode reset (continuing task; we just truncate) ----
        if episode_steps >= episode_len:
            episode_count += 1
            episode_reward = 0.0
            episode_steps = 0
            state = env.reset()
            agent.reset_state_sequence(state)

        # ---- Learning ----
        diagnostics: dict[str, float] = {}
        if step > learning_starts and step % train_every == 0:
            diagnostics = agent.learn()

        # ---- Target network update ----
        agent.maybe_update_target(step)

        agent.increment_step()

        # ---- Logging ----
        now = time.time()
        if step % 1000 == 0 or step == n_steps:
            elapsed = now - t_start
            steps_per_sec = step / max(elapsed, 1e-9)
            eps = agent.epsilon(step - 1)
            mean_r = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            mean_S = float(np.mean(recent_S)) if recent_S else 0.0
            mean_pcoll = float(np.mean(recent_pcoll)) if recent_pcoll else 0.0
            line = (
                f"step {step:>7d}/{n_steps} | "
                f"eps={eps:.3f} | "
                f"r1000={mean_r:+.4f} | "
                f"S1000={mean_S:.4f} | "
                f"pcoll1000={mean_pcoll:.4f} | "
                f"{steps_per_sec:.1f} steps/s"
            )
            if diagnostics:
                line += (f" | loss={diagnostics.get('loss', float('nan')):.4f} "
                         f"| q_mean={diagnostics.get('q_mean', float('nan')):+.3f}")
            print(line)

        # ---- Checkpointing ----
        if step % checkpoint_every == 0 or step == n_steps:
            ckpt_path = ckpt_dir / f"drqn_mha_seed{seed}_step{step}.pt"
            torch.save({
                "state_dict": agent.state_dict(),
                "config": cfg,
                "step": step,
                "seed": seed,
            }, ckpt_path)
            print(f"  [CKPT] saved → {ckpt_path}")

    elapsed = time.time() - t_start
    final_ckpt = ckpt_dir / f"drqn_mha_seed{seed}_final.pt"
    torch.save({
        "state_dict": agent.state_dict(),
        "config": cfg,
        "step": n_steps,
        "seed": seed,
    }, final_ckpt)
    print("-" * 78)
    print(f"[DONE] Training finished in {elapsed/60:.2f} min")
    print(f"[DONE] Final checkpoint: {final_ckpt}")
    return str(final_ckpt)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Train DRQN-MHA (paper reproduction).")
    ap.add_argument("--config", type=str, default="configs/default.yaml",
                    help="Path to YAML config (default: configs/default.yaml)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed (default: 42)")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override total training steps (default: from config)")
    ap.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Override checkpoint directory")
    ap.add_argument("--device", type=str, default=None,
                    choices=["cuda", "cpu"],
                    help="Override device (cuda/cpu)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PACKAGE_ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg["device"] = args.device

    set_seed(args.seed)
    train(cfg, seed=args.seed, total_steps=args.steps,
          checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
