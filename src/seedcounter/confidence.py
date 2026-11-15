"""Seed confidence: a logistic model over measured seed features.

The coefficients live in the config (confidence.seed_model) and are fitted against the
ground-truth annotations by scripts/calibrate.py, so a seed's confidence approximates the
probability that it is a real, separately counted seed.
"""
from __future__ import annotations

import math

import numpy as np

MODEL_FEATURES = ["contrast", "size_deviation", "solidity_deficit", "centerline_cross", "split", "end_proximity"]


def model_inputs(features: dict) -> dict:
    """Raw measured features -> the transformed inputs the model uses."""
    return {
        "contrast": min(features.get("contrast", 1.0), 1.5),
        "size_deviation": abs(math.log(max(features.get("area_ratio", 1.0), 1e-3))),
        "solidity_deficit": 1.0 - features.get("solidity", 1.0),
        "centerline_cross": features.get("centerline_cross", 0.0),
        "split": float(features.get("split", 0.0)),
        "end_proximity": math.exp(-features.get("end_distance", 10.0)),
    }


def seed_confidence(features: dict, model: dict) -> float:
    x = model_inputs(features)
    z = model["intercept"] + sum(model[k] * x[k] for k in MODEL_FEATURES)
    return float(1.0 / (1.0 + math.exp(-max(min(z, 30), -30))))


def fit_logistic(rows: list[dict], labels: list[int], l2: float = 1.0, iters: int = 4000) -> dict:
    """Small L2-regularised logistic regression (numpy only) -> seed_model dict."""
    X = np.array([[model_inputs(r)[k] for k in MODEL_FEATURES] for r in rows], float)
    y = np.array(labels, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    # class weights so the rare false detections are not ignored
    pos = max(y.mean(), 1e-3)
    sw = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-3))
    lr = 0.1
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        grad = Z.T @ (sw * (p - y)) / len(y) + l2 * np.r_[0, w[1:]] / len(y)
        w -= lr * grad
    coef = w[1:] / sd
    intercept = w[0] - float(np.sum(coef * mu))
    # undo the class re-weighting so outputs are probabilities at the real base rate
    intercept += math.log(pos / max(1 - pos, 1e-3))
    model = {"intercept": round(float(intercept), 4)}
    model.update({k: round(float(c), 4) for k, c in zip(MODEL_FEATURES, coef)})
    return model
