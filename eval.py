
from __future__ import annotations

import argparse
import sys
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
from baselines.beb import BEBPolicy
from train import build_phy, build_env, build_agent


def load_checkpoint(path: str, device: torch.device) -> tuple[DRQNAgent, dict]:
    """Load a checkpoint and return the agent + config."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    agent = build_agent(cfg, device)
    agent.load_state_dict(ckpt["state_dict"])
    agent.online.eval()
    return agent, cfg


def evaluate_drqn(agent: DRQNAgent, env: WlanEnv, n_steps: int,
                  n_init: int, ru: int) -> dict[str, float]:
    """Evaluate the DRQN-MHA policy for `n_steps` starting at N=`n_init`."""
    state = env.reset(n_init=n_init, r=ru)
    agent.reset_state_sequence(state)
    agent.online.eval()

    rewards, S_vals, pcoll_vals, dhol_vals, coll_count = [], [], [], [], 0
    cw_history = []
    with torch.no_grad():
        for _ in range(n_steps):
            action = agent.observe_and_act(state, eval_mode=True)
            next_state, reward, _, info = env.step(action)
            rewards.append(reward)
            S_vals.append(info["S"])
            pcoll_vals.append(info["p_coll"])
            dhol_vals.append(info["D_hol"])
            cw_history.append(info["CW"])
            if info["collision"]:
                coll_count += 1
            state = next_state
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_S": float(np.mean(S_vals)),
        "std_S": float(np.std(S_vals)),
        "mean_pcoll": float(np.mean(pcoll_vals)),
        "mean_dhol": float(np.mean(dhol_vals)),
        "collision_rate": coll_count / n_steps,
        "mean_CW": float(np.mean(cw_history)),
    }


def evaluate_beb(env: WlanEnv, n_steps: int, n_init: int, ru: int) -> dict[str, float]:
    """Evaluate the BEB baseline."""
    beb = BEBPolicy(cw_min=env.phy.cw_min, cw_max=env.phy.cw_max,
                    action_space=env.action_space)
    state = env.reset(n_init=n_init, r=ru)
    beb.reset()

    rewards, S_vals, pcoll_vals, dhol_vals, coll_count = [], [], [], [], 0
    for _ in range(n_steps):
        action = beb.act()
        next_state, reward, _, info = env.step(action)
        beb.update(info["collision"])
        rewards.append(reward)
        S_vals.append(info["S"])
        pcoll_vals.append(info["p_coll"])
        dhol_vals.append(info["D_hol"])
        if info["collision"]:
            coll_count += 1
        state = next_state
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_S": float(np.mean(S_vals)),
        "std_S": float(np.std(S_vals)),
        "mean_pcoll": float(np.mean(pcoll_vals)),
        "mean_dhol": float(np.mean(dhol_vals)),
        "collision_rate": coll_count / n_steps,
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Evaluate a trained DRQN-MHA checkpoint.")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to .pt checkpoint")
    ap.add_argument("--ru", type=int, default=6, choices=[3, 6, 9],
                    help="Number of OFDMA RUs (default: 6, matches paper Table V)")
    ap.add_argument("--eval-steps", type=int, default=1000,
                    help="Evaluation steps per density class (default: 1000)")
    ap.add_argument("--density-class", type=str, default=None,
                    help="Evaluate only this class (default: all six)")
    ap.add_argument("--device", type=str, default=None,
                    choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=12345,
                    help="Seed for env RNG during evaluation")
    args = ap.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[INFO] Device: {device}")

    agent, cfg = load_checkpoint(args.checkpoint, device)
    phy = build_phy(cfg)

    package_root = Path(__file__).resolve().parent
    cache_dir = str(package_root / ".cache")

    # Density classes from paper Table VI
    classes = [
        ("Very Sparse",     1,   50),
        ("Sparse",          50,  150),
        ("Moderate",        150, 250),
        ("Dense",           250, 350),
        ("Very Dense",      350, 450),
        ("Extremely Dense", 450, 512),
    ]
    if args.density_class:
        classes = [c for c in classes if c[0].lower() == args.density_class.lower()]
        if not classes:
            print(f"[ERROR] density class '{args.density_class}' not found.")
            sys.exit(1)

    print(f"[INFO] Evaluating on r={args.ru}, {args.eval_steps} steps per class")
    print("=" * 90)
    print(f"{'Density Class':<20} {'N_mid':>6} "
          f"{'DRQN-MHA S':>12} {'BEB S':>10} {'Δ vs BEB':>10} "
          f"{'DRQN pcoll':>12} {'BEB pcoll':>11}")
    print("-" * 90)

    for name, n_min, n_max in classes:
        env = build_env(cfg, phy, cache_dir, seed=args.seed)
        n_mid = (n_min + n_max) // 2
        drqn = evaluate_drqn(agent, env, args.eval_steps, n_init=n_mid, ru=args.ru)
        env2 = build_env(cfg, phy, cache_dir, seed=args.seed)
        beb = evaluate_beb(env2, args.eval_steps, n_init=n_mid, ru=args.ru)
        delta_pct = (drqn["mean_S"] - beb["mean_S"]) / max(beb["mean_S"], 1e-9) * 100
        print(f"{name:<20} {n_mid:>6d} "
              f"{drqn['mean_S']:>12.4f} {beb['mean_S']:>10.4f} "
              f"{delta_pct:>+9.2f}% "
              f"{drqn['mean_pcoll']:>12.4f} {beb['mean_pcoll']:>11.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
