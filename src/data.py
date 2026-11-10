import hashlib

import numpy as np
import polars as pl


SEED = 20261110
ASSETS = ["growth", "value", "international", "treasury", "credit", "gold"]
LOOKBACK, HORIZON = 12, 5
FOLDS = [(260, 340), (340, 420), (420, 500)]


def market(seed=SEED, days=505):
    """Correlated synthetic daily returns with predeclared volatility regimes."""
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(days) * .007
    regime = np.ones(days)
    regime[170:200], regime[370:395] = 2.3, 2.6
    loadings = np.array([1.15, .85, 1., -.20, .25, .05])
    idio = np.array([.007, .005, .006, .003, .0035, .006])
    returns = regime[:, None] * (factor[:, None]*loadings + rng.standard_normal((days, 6))*idio)
    returns += np.array([.00035, .0003, .00025, .0001, .00016, .00013])
    returns[370:375, :3] -= .018
    # Mild mean reversion, observable only through trailing features.
    for day in range(1, days):
        returns[day] -= .10*returns[day-1]
    dates = np.busday_offset(np.datetime64("2024-01-02"), np.arange(days)).astype(str)
    frame = pl.DataFrame({"day": np.repeat(np.arange(days), 6), "date": np.repeat(dates, 6),
                          "asset": np.tile(ASSETS, days), "return": returns.reshape(-1)})
    return frame, returns


def features(frame):
    return frame.sort(["asset", "day"]).with_columns(
        pl.col("return").rolling_mean(window_size=5, min_samples=5).over("asset").alias("momentum_5"),
        pl.col("return").rolling_std(window_size=20, min_samples=20, ddof=1).over("asset").alias("vol_20")
    ).sort(["day", "asset"])


def samples(frame, returns):
    f = features(frame)
    # Preserve explicit asset order; alphabetical sorting must not reorder targets.
    values = np.stack([f.filter(pl.col("asset")==asset).sort("day").select(
        ["return", "momentum_5", "vol_20"]).to_numpy() for asset in ASSETS], axis=1)
    x, y, days, assets = [], [], [], []
    for day in range(20+LOOKBACK-1, len(returns)-HORIZON):
        for asset in range(6):
            seq = values[day-LOOKBACK+1:day+1, asset]
            identity = np.tile(np.eye(6)[asset], (LOOKBACK, 1))
            future = returns[day+1:day+1+HORIZON, asset]
            x.append(np.column_stack([seq, identity]))
            y.append([(np.prod(1+future)-1)*100, np.std(future, ddof=1)*np.sqrt(HORIZON)*100])
            days.append(day)
            assets.append(asset)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), np.array(days), np.array(assets)


def training_mask(days, test_start):
    # Last training label endpoint must be strictly before the first test feature date.
    return days + HORIZON < test_start


def digest(returns):
    return hashlib.sha256(returns.astype("<f8").tobytes()).hexdigest()
