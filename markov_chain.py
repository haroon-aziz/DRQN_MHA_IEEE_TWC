
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class PhyParams:
    """Container for IEEE 802.11ax PHY/MAC parameters (paper Table I)."""
    payload_bits: int = 8184
    phy_header_bits: int = 128
    mac_header_bits: int = 272
    channel_bit_rate_mbps: float = 1.0
    slot_time_us: float = 50.0
    cw_min: int = 32
    cw_max: int = 1024
    max_backoff_stages: int = 5  # m

def _slot_time_s(p: PhyParams) -> float:
"
    return p.slot_time_us * 1e-6


def _payload_time_s(p: PhyParams) -> float:
  
    total_bits = p.payload_bits + p.phy_header_bits + p.mac_header_bits
    return total_bits / (p.channel_bit_rate_mbps * 1e6)


def _collision_time_s(p: PhyParams) -> float:
    
    total_bits = p.payload_bits + p.phy_header_bits + p.mac_header_bits
    return total_bits / (p.channel_bit_rate_mbps * 1e6)


def _cw_at_stage(stage: int, p: PhyParams) -> int:
   ""
    return min((2 ** stage) * p.cw_min, p.cw_max)


def tau_eq(p_coll: float, cw: int, p: PhyParams) -> float:
 
    m = p.max_backoff_stages
    # Numerical guards
    p_coll = min(max(p_coll, 1e-9), 0.5 - 1e-6)

    num = 2.0 * (1.0 - 2.0 * p_coll)
    term1 = (1.0 - 2.0 * p_coll) * (cw + 1)
    term2 = p_coll * cw * (1.0 - (2.0 * p_coll) ** m)
    den = term1 + term2
    if den <= 0:
        return 1.0
    tau = num / den
    return float(min(max(tau, 1e-9), 1.0))


def p_coll_eq(tau: float, n_per_ru: float) -> float:
 
    if n_per_ru <= 1:
        return 0.0
    exponent = n_per_ru - 1.0
    return float(1.0 - (1.0 - tau) ** exponent)


def fixed_point(cw: int, n: int, r: int, p: PhyParams,
                max_iter: int = 200, tol: float = 1e-10) -> tuple[float, float]:
  
    n_per_ru = max(n / r, 1.0)
    tau = 1.0 / (cw + 1.0)
    p_coll = 0.0
    for _ in range(max_iter):
        tau_new = tau_eq(p_coll, cw, p)
        p_coll_new = p_coll_eq(tau_new, n_per_ru)
        if abs(tau_new - tau) + abs(p_coll_new - p_coll) < tol:
            tau, p_coll = tau_new, p_coll_new
            break
        tau, p_coll = tau_new, p_coll_new
    return tau, p_coll


def saturation_throughput(tau: float, p_coll: float, n: int, r: int,
                          p: PhyParams) -> float:
   
    n_per_ru = max(n / r, 1.0)
  
    sigma = 1.0

    total_bits = p.payload_bits + p.phy_header_bits + p.mac_header_bits
    packet_time_s = total_bits / (p.channel_bit_rate_mbps * 1e6)
    ts = packet_time_s / (p.slot_time_us * 1e-6) 
 
    tc = ts


    p_tr = 1.0 - (1.0 - tau) ** n_per_ru                 
    if p_tr <= 0:
        return 0.0
    # P(success | ≥1 tx) = N/r · τ · (1−τ)^(N/r − 1) / P_tr
    p_s_given_tr = n_per_ru * tau * (1.0 - tau) ** (n_per_ru - 1.0) / p_tr
    p_s_given_tr = float(np.clip(p_s_given_tr, 0.0, 1.0))

    # Bianchi's formula (per-RU)
    num = p_tr * p_s_given_tr * ts
    den = (p_tr * p_s_given_tr * ts
           + p_tr * (1.0 - p_s_given_tr) * tc
           + (1.0 - p_tr) * sigma)
    if den <= 0:
        return 0.0
    return float(num / den)


def hol_delay(tau: float, p_coll: float, n: int, r: int,
              p: PhyParams) -> float:
 
    if tau <= 0:
        return 1e6
    if p_coll <= 0:
        # No collision → expected delay ≈ 1/τ slots
        return float(1.0 / tau)
   
    expected_attempts = 1.0 / max(1.0 - p_coll, 1e-6)
    expected_slots_per_attempt = (1.0 / tau)  # mean backoff + 1 attempt
    return float(expected_attempts * expected_slots_per_attempt)


def compute_metrics(cw: int, n: int, r: int, p: PhyParams
                    ) -> tuple[float, float, float, float]:
   
    tau, p_coll = fixed_point(cw, n, r, p)
    S = saturation_throughput(tau, p_coll, n, r, p)
    D = hol_delay(tau, p_coll, n, r, p)
    return tau, p_coll, S, D
