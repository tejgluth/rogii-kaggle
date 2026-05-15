# Phase 3: Feature Engineering

Your experiment spec tells you which features to add. Read the spec carefully.
Build on the best previous experiment unless told otherwise.

## Feature Engineering Ideas (Pick From These Based on Spec)

### GR Signal Features
```python
# Rolling statistics at multiple windows
for w in [5, 10, 20, 30, 50, 100]:
    df[f"gr_mean_{w}"] = df.groupby("well_id")["gr"].transform(
        lambda x: x.rolling(w, min_periods=1).mean())
    df[f"gr_std_{w}"] = df.groupby("well_id")["gr"].transform(
        lambda x: x.rolling(w, min_periods=1).std())
    df[f"gr_min_{w}"] = df.groupby("well_id")["gr"].transform(
        lambda x: x.rolling(w, min_periods=1).min())
    df[f"gr_max_{w}"] = df.groupby("well_id")["gr"].transform(
        lambda x: x.rolling(w, min_periods=1).max())

# GR gradient (rate of change)
df["gr_diff_1"] = df.groupby("well_id")["gr"].diff(1)
df["gr_diff_5"] = df.groupby("well_id")["gr"].diff(5)

# GR percentile rank within well
df["gr_pct_rank"] = df.groupby("well_id")["gr"].transform(
    lambda x: x.rank(pct=True))

# GR z-score within well
df["gr_zscore"] = df.groupby("well_id")["gr"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8))
```

### Typewell Correlation Features (Most Powerful)
```python
from scipy.signal import correlate
from scipy.spatial.distance import cdist

def compute_typewell_correlation(well_gr, typewell_gr, max_lag=50):
    """Cross-correlation between well GR and Typewell GR at various lags."""
    corr = correlate(well_gr, typewell_gr, mode="full")
    lags = np.arange(-max_lag, max_lag + 1)
    # Extract correlation values at each lag
    center = len(corr) // 2
    corr_at_lags = corr[center - max_lag: center + max_lag + 1]
    best_lag = lags[np.argmax(corr_at_lags)]
    best_corr = corr_at_lags[np.argmax(corr_at_lags)]
    return best_lag, best_corr, corr_at_lags

# For each well, compute at every depth sample:
# - The best-matching typewell lag in a sliding window
# - The correlation coefficient at that lag
# This is the core TVT signal!
```

### DTW Features (Dynamic Time Warping)
```python
# DTW distance between local GR window and typewell GR
# Use fastdtw or tslearn for speed
# Window: 20-50 samples each side
# Compute at lag offsets: -30 to +30 ft
# The lag with minimum DTW distance estimates TVT shift
from fastdtw import fastdtw

def dtw_lag_features(well_gr, typewell_gr, window=30, lag_range=40):
    dtw_distances = []
    for lag in range(-lag_range, lag_range + 1):
        shifted_typewell = np.roll(typewell_gr, lag)
        dist, _ = fastdtw(well_gr[:window], shifted_typewell[:window])
        dtw_distances.append(dist)
    return np.array(dtw_distances)
```

### Trajectory Features
```python
# Dog-leg severity (DLS) — rate of direction change
df["dls"] = np.sqrt(
    df.groupby("well_id")["inclination"].diff()**2 +
    df.groupby("well_id")["azimuth"].diff()**2
)

# Structural dip estimate from TVD change rate
df["tvd_gradient"] = df.groupby("well_id")["tvd"].diff()

# Cumulative lateral distance
df["lateral_dist"] = df.groupby("well_id")["md"].transform(
    lambda x: x - x.iloc[0])

# XY displacement from well heel
df["xy_dist"] = np.sqrt(
    (df.groupby("well_id")["x"].transform(lambda x: x - x.iloc[0]))**2 +
    (df.groupby("well_id")["y"].transform(lambda x: x - x.iloc[0]))**2
)
```

### Lag/Lead Features
```python
# Previous TVT values can be used as features (careful: no leakage in CV)
# Only use these after verifying no leakage with GroupKFold
for lag in [1, 3, 5, 10, 20]:
    df[f"gr_lag_{lag}"] = df.groupby("well_id")["gr"].shift(lag)
    df[f"gr_lead_{lag}"] = df.groupby("well_id")["gr"].shift(-lag)
```

### Target Encoding
```python
# Well-level GR statistics as context
well_stats = df.groupby("well_id")["gr"].agg(["mean", "std", "min", "max"])
df = df.join(well_stats, on="well_id", rsuffix="_well")
```

## Instructions

1. Read the experiment spec to know which features to add
2. Build on the feature set from `base_experiment` (load that code from src/)
3. Add ONLY the features specified — don't add everything at once
4. Run with LightGBM GPU unless spec says otherwise
5. Compare CV RMSE to base_experiment CV RMSE
6. In result JSON notes, explain: did the feature help? Why or why not?
7. Add successful feature code to `src/features.py`
