"""
IEEE 802.11ax OFDMA Markov-chain solver.

Implements the Bianchi/Lanante fixed-point equations referenced in
Section III-B of the paper:

    τ = 2(1 − 2p) / [(1 − 2p)(CW_min + 1) + p · CW_min · (1 − (2p)^m)]   (1)
    p = 1 − (1 − τ)^(N/r − 1)                                            (2)

and the closed-form saturation throughput S, collision probability P_coll,
and head-of-line delay D_hol as functions of (N, r, CW).

References
----------
* Bianchi, "Performance Analysis of the IEEE 802.11 DCF," JSAC 2000. [6]
* Lanante et al., "Fuzzy Logic Based CW Optimization for OFDMA-Based
  IEEE 802.11ax WLANs," TWC 2021. [1]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Physical-layer timing constants (Table I).
# All times are expressed in slot-time units (δ = 50 μs) so S, P_coll, D_hol
# are dimensionless or in slots, matching the paper's reporting convention.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core closed-form expressions
# ---------------------------------------------------------------------------

def _slot_time_s(p: PhyParams) -> float:
    """Return slot time δ in seconds."""
    return p.slot_time_us * 1e-6


def _payload_time_s(p: PhyParams) -> float:
    """Successful-packet payload transmission time (s)."""
    total_bits = p.payload_bits + p.phy_header_bits + p.mac_header_bits
    return total_bits / (p.channel_bit_rate_mbps * 1e6)


def _collision_time_s(p: PhyParams) -> float:
    """Collision time = payload + headers (no ACK); same as payload time here."""
    total_bits = p.payload_bits + p.phy_header_bits + p.mac_header_bits
    return total_bits / (p.channel_bit_rate_mbps * 1e6)


def _cw_at_stage(stage: int, p: PhyParams) -> int:
    """CW_i = min(2^i · CW_min, CW_max) (Sec III-B)."""
    return min((2 ** stage) * p.cw_min, p.cw_max)


def tau_eq(p_coll: float, cw: int, p: PhyParams) -> float:
    """Per-STA transmission probability τ from Eq. (1).

    Parameters
    ----------
    p_coll : float
        Conditional collision probability p (given a transmission attempt).
    cw : int
        Contention window value used by the agent (treated as CW_min in the
        Markov chain — i.e. the agent sets the base CW and the chain still
        has m backoff stages above it).
    p : PhyParams
        PHY/MAC parameters.

    Returns
    -------
    float
        τ ∈ (0, 1].

    Notes
    -----
    If p == 0.5 the denominator degenerates; we clamp p to (0, 0.5) before
    evaluating. Numerical guards also handle the p → 0 and p → 0.5 limits.
    """
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
    """Conditional collision probability p from Eq. (2).

    Parameters
    ----------
    tau : float
        Per-STA transmission probability.
    n_per_ru : float
        Number of contending STAs per RU = N / r.

    Returns
    -------
    float
        p ∈ [0, 1).
    """
    if n_per_ru <= 1:
        return 0.0
    exponent = n_per_ru - 1.0
    return float(1.0 - (1.0 - tau) ** exponent)


def fixed_point(cw: int, n: int, r: int, p: PhyParams,
                max_iter: int = 200, tol: float = 1e-10) -> tuple[float, float]:
    """Solve the (τ, p) fixed-point system (Eqs. 1 & 2) by iteration.

    Parameters
    ----------
    cw : int
        Base CW value (one of {32, 64, 128, 256, 512, 1024}).
    n : int
        Number of active STAs (1–512).
    r : int
        Number of OFDMA RUs (∈ {3, 6, 9}).
    p : PhyParams
        PHY parameters.
    max_iter : int
        Maximum fixed-point iterations.
    tol : float
        Convergence tolerance on |Δτ| + |Δp|.

    Returns
    -------
    (tau, p_coll) : tuple[float, float]
    """
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


# ---------------------------------------------------------------------------
# Saturation throughput, collision probability, head-of-line delay
# ---------------------------------------------------------------------------

def saturation_throughput(tau: float, p_coll: float, n: int, r: int,
                          p: PhyParams) -> float:
    """Normalized saturation throughput S ∈ [0, 1] (Bianchi, extended to OFDMA).

    Bianchi's normalized throughput for one channel:

        S = (P_tr · P_s · T_s) /
            (P_tr · P_s · T_s + P_tr · (1 − P_s) · T_c + (1 − P_tr) · σ)

    For OFDMA-RA with r RUs, each RU is a separate contention domain with
    N/r contending STAs. The normalized system throughput equals the per-RU
    S (because each RU carries 1/r of the channel capacity):

        S_system = S_per_RU ∈ [0, 1]

    Parameters
    ----------
    tau, p_coll : float
        Fixed-point solution.
    n : int
        Number of STAs.
    r : int
        Number of RUs.
    p : PhyParams
        PHY parameters.

    Returns
    -------
    float
        Normalized throughput S ∈ [0, 1].
    """
    n_per_ru = max(n / r, 1.0)
    # All times in slot-time units (δ = 1)
    sigma = 1.0
    # Total packet transmission time = (PHY+MAC+Payload) bits / rate / δ
    total_bits = p.payload_bits + p.phy_header_bits + p.mac_header_bits
    packet_time_s = total_bits / (p.channel_bit_rate_mbps * 1e6)
    ts = packet_time_s / (p.slot_time_us * 1e-6)  # in slot units
    # Basic access (no RTS/CTS): collision time ≈ full packet time
    tc = ts

    # Per-RU probabilities
    p_tr = 1.0 - (1.0 - tau) ** n_per_ru                 # ≥1 STA transmits
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
    """Head-of-line delay D_hol in slot-time units.

    Approximation consistent with Bianchi's framework:

        D_hol ≈ (1 − p) / (p · τ)  · σ   (expected attempts until success)

    For OFDMA we use the per-RU contention group size N/r.

    Parameters
    ----------
    tau, p_coll : float
        Fixed-point solution.
    n, r, p : see `saturation_throughput`.

    Returns
    -------
    float
        D_hol in slot-time units (δ = 50 μs each).
    """
    if tau <= 0:
        return 1e6
    if p_coll <= 0:
        # No collision → expected delay ≈ 1/τ slots
        return float(1.0 / tau)
    # Expected number of attempts until success: 1/(1-p)
    # Expected slots per attempt ≈ (CW-1)/2 + 1
    # Combine: D_hol ≈ E[attempts] · E[slots/attempt]
    expected_attempts = 1.0 / max(1.0 - p_coll, 1e-6)
    expected_slots_per_attempt = (1.0 / tau)  # mean backoff + 1 attempt
    return float(expected_attempts * expected_slots_per_attempt)


def compute_metrics(cw: int, n: int, r: int, p: PhyParams
                    ) -> tuple[float, float, float, float]:
    """One-shot computation of (τ, p_coll, S, D_hol) for a (N, r, CW) triple.

    Parameters
    ----------
    cw : int
        Base contention window.
    n : int
        Number of active STAs.
    r : int
        Number of RUs.
    p : PhyParams
        PHY parameters.

    Returns
    -------
    (tau, p_coll, S, D_hol) : tuple[float, float, float, float]
        τ and p are dimensionless probabilities; S is normalized throughput
        in [0, 1]; D_hol is in slot-time units.
    """
    tau, p_coll = fixed_point(cw, n, r, p)
    S = saturation_throughput(tau, p_coll, n, r, p)
    D = hol_delay(tau, p_coll, n, r, p)
    return tau, p_coll, S, D
