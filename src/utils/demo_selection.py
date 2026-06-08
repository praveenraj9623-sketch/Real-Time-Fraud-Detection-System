"""Lightweight demo-row selection helpers.

This module intentionally avoids FastAPI, SHAP, PySpark, and model imports so
unit tests can exercise demo sampling quickly.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.utils.risk import DEFAULT_SAVE_THRESHOLD, clip_probability


def _sample_without_replacement(
    rng: np.random.Generator,
    candidates: np.ndarray,
    take: int,
) -> list[int]:
    if take <= 0 or len(candidates) == 0:
        return []
    actual_take = min(take, len(candidates))
    return rng.choice(candidates, size=actual_take, replace=False).astype(int).tolist()


def _amount_bucket_pools(candidates: np.ndarray, amount_values: np.ndarray) -> list[np.ndarray]:
    """Split candidates into low/medium/high amount pools when possible."""
    if len(candidates) == 0:
        return []

    finite_amounts = np.nan_to_num(amount_values[candidates], nan=0.0, posinf=0.0, neginf=0.0)
    non_tiny = candidates[finite_amounts > 1.0]
    source = non_tiny if len(non_tiny) >= 3 else candidates
    source_amounts = np.nan_to_num(amount_values[source], nan=0.0, posinf=0.0, neginf=0.0)

    if len(source) < 3 or len(np.unique(source_amounts)) < 2:
        return [source]

    low_cutoff, high_cutoff = np.nanpercentile(source_amounts, [33, 66])
    pools = [
        source[source_amounts <= low_cutoff],
        source[(source_amounts > low_cutoff) & (source_amounts <= high_cutoff)],
        source[source_amounts > high_cutoff],
    ]
    return [pool for pool in pools if len(pool) > 0]


def select_demo_indices(
    probabilities: Iterable[float],
    *,
    max_alerts: int,
    save_threshold: float = DEFAULT_SAVE_THRESHOLD,
    model_threshold: float = 0.5,
    amounts: Iterable[float] | None = None,
    random_state: int = 42,
) -> list[int]:
    """Choose a varied set of alert rows across review, flagged, and blocked bands.

    When amounts are supplied, part of each band is sampled from higher-amount
    rows so the demo does not over-index on tiny $1 transactions.
    """
    probs = np.clip(np.asarray(list(probabilities), dtype=float), 0.0, 1.0)
    if probs.size == 0 or max_alerts <= 0:
        return []

    amount_values = None
    if amounts is not None:
        amount_values = np.asarray(list(amounts), dtype=float)
        if amount_values.shape != probs.shape:
            amount_values = None

    save_at = clip_probability(save_threshold)
    block_at = max(clip_probability(model_threshold), 0.50)
    rng = np.random.default_rng(random_state)

    bands = [
        np.where((probs >= save_at) & (probs < 0.50))[0],
        np.where((probs >= 0.50) & (probs < block_at))[0],
        np.where(probs >= block_at)[0],
    ]
    quotas = [
        max(1, int(round(max_alerts * 0.35))),
        max(1, int(round(max_alerts * 0.30))),
        max(1, max_alerts - int(round(max_alerts * 0.65))),
    ]

    selected: list[int] = []
    for band, quota in zip(bands, quotas):
        if len(band) == 0:
            continue
        quota = min(quota, max_alerts - len(selected))
        if quota <= 0:
            break

        band_selected = 0
        if amount_values is not None:
            for pool in _amount_bucket_pools(band, amount_values):
                if band_selected >= quota or len(selected) >= max_alerts:
                    break
                pool = np.array([idx for idx in pool if int(idx) not in set(selected)], dtype=int)
                take = max(1, quota // 3)
                sampled = _sample_without_replacement(rng, pool, min(take, quota - band_selected))
                selected.extend(sampled)
                band_selected += len(sampled)

        remaining_pool = np.array([idx for idx in band if int(idx) not in set(selected)], dtype=int)
        if amount_values is not None:
            amount_pool = np.nan_to_num(amount_values[remaining_pool], nan=0.0, posinf=0.0, neginf=0.0)
            non_tiny_pool = remaining_pool[amount_pool > 1.0]
            if len(non_tiny_pool) > 0:
                remaining_pool = non_tiny_pool
        remaining_take = min(quota - band_selected, max_alerts - len(selected))
        selected.extend(_sample_without_replacement(rng, remaining_pool, remaining_take))

    if len(selected) < max_alerts:
        candidates = np.where(probs >= save_at)[0]
        if amount_values is not None and len(candidates) > 0:
            amount_candidates = np.nan_to_num(amount_values[candidates], nan=0.0, posinf=0.0, neginf=0.0)
            varied_candidates = candidates[amount_candidates > 1.0]
            if len(varied_candidates) > 0:
                candidates = varied_candidates
        remaining = np.array([idx for idx in candidates if int(idx) not in set(selected)], dtype=int)
        selected.extend(_sample_without_replacement(rng, remaining, max_alerts - len(selected)))

    if len(selected) < max_alerts:
        candidates = np.where(probs >= save_at)[0]
        remaining = np.array([idx for idx in candidates if int(idx) not in set(selected)], dtype=int)
        selected.extend(_sample_without_replacement(rng, remaining, max_alerts - len(selected)))

    unique_selected = list(dict.fromkeys(selected))
    if amount_values is not None:
        def sort_key(idx: int) -> tuple[float, float, float]:
            amount = float(np.nan_to_num(amount_values[idx], nan=0.0, posinf=0.0, neginf=0.0))
            return (1.0 if amount > 1.0 else 0.0, float(probs[idx]), amount)

        unique_selected = sorted(unique_selected, key=sort_key, reverse=True)
    else:
        unique_selected = sorted(unique_selected, key=lambda idx: probs[idx], reverse=True)
    return unique_selected[:max_alerts]
