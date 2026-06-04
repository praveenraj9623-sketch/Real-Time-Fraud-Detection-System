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

        high_amount_take = 0
        if amount_values is not None and len(band) > 2:
            band_amounts = amount_values[band]
            high_cutoff = np.nanpercentile(band_amounts, 70)
            high_amount_pool = band[band_amounts >= high_cutoff]
            high_amount_take = max(1, quota // 3)
            selected.extend(_sample_without_replacement(rng, high_amount_pool, high_amount_take))

        remaining_pool = np.array([idx for idx in band if int(idx) not in set(selected)], dtype=int)
        remaining_take = min(quota - high_amount_take, max_alerts - len(selected))
        selected.extend(_sample_without_replacement(rng, remaining_pool, remaining_take))

    if len(selected) < max_alerts:
        candidates = np.where(probs >= save_at)[0]
        if amount_values is not None and len(candidates) > 0:
            high_cutoff = np.nanpercentile(amount_values[candidates], 70)
            candidates = candidates[amount_values[candidates] >= high_cutoff]
        remaining = np.array([idx for idx in candidates if int(idx) not in set(selected)], dtype=int)
        selected.extend(_sample_without_replacement(rng, remaining, max_alerts - len(selected)))

    if len(selected) < max_alerts:
        candidates = np.where(probs >= save_at)[0]
        remaining = np.array([idx for idx in candidates if int(idx) not in set(selected)], dtype=int)
        selected.extend(_sample_without_replacement(rng, remaining, max_alerts - len(selected)))

    selected = sorted(set(selected), key=lambda idx: probs[idx], reverse=True)
    return selected[:max_alerts]

