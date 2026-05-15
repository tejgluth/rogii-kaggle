"""
Feature Engineering
===================
All feature engineering functions. Claude Code adds new functions here
as experiments succeed. Never delete existing functions.
"""

from pathlib import Path

try:
    import cudf as pd
    GPU = True
except ImportError:
    import pandas as pd
    GPU = False

import numpy as np


# ============================================================
# SHARED HELPERS
# ============================================================

def _rolling_slope_by_group(df, value_col, x_col, group_col, window):
    """Causal rolling linear-regression slope of value_col vs x_col."""
    slopes = np.zeros(len(df), dtype=np.float32)

    for _, idx in df.groupby(group_col, sort=False).groups.items():
        values = df.loc[idx, value_col].to_numpy(dtype=np.float64, copy=False)
        x = df.loc[idx, x_col].to_numpy(dtype=np.float64, copy=False)
        n = len(values)
        if n == 0:
            continue

        valid = np.isfinite(values) & np.isfinite(x)
        y = np.where(valid, values, 0.0)
        xv = np.where(valid, x, 0.0)
        count = valid.astype(np.float64)

        c_n = np.r_[0.0, np.cumsum(count)]
        c_x = np.r_[0.0, np.cumsum(xv)]
        c_y = np.r_[0.0, np.cumsum(y)]
        c_xx = np.r_[0.0, np.cumsum(xv * xv)]
        c_xy = np.r_[0.0, np.cumsum(xv * y)]

        end = np.arange(1, n + 1)
        start = np.maximum(0, end - window)
        sum_n = c_n[end] - c_n[start]
        sum_x = c_x[end] - c_x[start]
        sum_y = c_y[end] - c_y[start]
        sum_xx = c_xx[end] - c_xx[start]
        sum_xy = c_xy[end] - c_xy[start]

        denom = (sum_n * sum_xx) - (sum_x * sum_x)
        numer = (sum_n * sum_xy) - (sum_x * sum_y)
        group_slopes = np.divide(
            numer,
            denom,
            out=np.zeros(n, dtype=np.float64),
            where=(sum_n >= 2) & (np.abs(denom) > 1e-12),
        )
        slopes[np.asarray(idx)] = group_slopes.astype(np.float32)

    return slopes


# ============================================================
# BASELINE FEATURES
# ============================================================

def add_depth_features(df):
    """Basic depth position features."""
    df = df.copy()
    # Distance from start of lateral per well
    df["depth_from_heel"] = df.groupby("well_id")["md"].transform(
        lambda x: x - x.min()
    )
    # Normalized depth position (0=heel, 1=toe)
    df["depth_fraction"] = df.groupby("well_id")["md"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )
    return df


def add_gr_rolling_features(df, windows=(5, 10, 20, 30, 50)):
    """Rolling statistics on GR log."""
    df = df.copy()
    if "gr" not in df.columns:
        print("WARNING: 'gr' column not found, skipping GR features")
        return df

    for w in windows:
        df[f"gr_mean_{w}"] = df.groupby("well_id")["gr"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"gr_std_{w}"] = df.groupby("well_id")["gr"].transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0)
        )
        df[f"gr_min_{w}"] = df.groupby("well_id")["gr"].transform(
            lambda x: x.rolling(w, min_periods=1).min()
        )
        df[f"gr_max_{w}"] = df.groupby("well_id")["gr"].transform(
            lambda x: x.rolling(w, min_periods=1).max()
        )
    return df


def add_gr_gradient_features(df, periods=(1, 3, 5, 10)):
    """GR rate of change features."""
    df = df.copy()
    for p in periods:
        df[f"gr_diff_{p}"] = df.groupby("well_id")["gr"].transform(
            lambda x: x.diff(p)
        ).fillna(0)
    return df


def add_gr_normalization_features(df):
    """GR percentile rank and z-score within each well."""
    df = df.copy()
    df["gr_pct_rank"] = df.groupby("well_id")["gr"].transform(
        lambda x: x.rank(pct=True)
    )
    df["gr_zscore"] = df.groupby("well_id")["gr"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    return df


def add_trajectory_features(df):
    """Dog-leg severity and structural dip features."""
    df = df.copy()
    if "inclination" in df.columns and "azimuth" in df.columns:
        inc_diff = df.groupby("well_id")["inclination"].transform(
            lambda x: x.diff().fillna(0)
        )
        az_diff = df.groupby("well_id")["azimuth"].transform(
            lambda x: x.diff().fillna(0)
        )
        df["dls"] = np.sqrt(inc_diff**2 + az_diff**2)

    if "tvd" in df.columns:
        df["tvd_gradient"] = df.groupby("well_id")["tvd"].transform(
            lambda x: x.diff().fillna(0)
        )
    return df


def add_well_context_features(df):
    """Well-level aggregate statistics as context features."""
    df = df.copy()
    if "gr" in df.columns:
        well_stats = df.groupby("well_id")["gr"].agg(["mean", "std", "min", "max"])
        well_stats.columns = ["well_gr_mean", "well_gr_std",
                              "well_gr_min", "well_gr_max"]
        if GPU:
            df = df.merge(well_stats, on="well_id", how="left")
        else:
            df = df.join(well_stats, on="well_id")
    return df


def add_gr_lag_lead_features(df, lags=(1, 3, 5, 10, 20)):
    """Past and future GR values as features."""
    df = df.copy()
    for lag in lags:
        df[f"gr_lag_{lag}"] = df.groupby("well_id")["gr"].transform(
            lambda x: x.shift(lag)
        ).fillna(method="bfill")
        df[f"gr_lead_{lag}"] = df.groupby("well_id")["gr"].transform(
            lambda x: x.shift(-lag)
        ).fillna(method="ffill")
    return df


# ============================================================
# ADVANCED FEATURES (added as experiments succeed)
# ============================================================

def _first_existing_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"None of these columns were found: {names}")


def _as_numpy(values):
    if hasattr(values, "to_pandas"):
        values = values.to_pandas()
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def _rolling_window_bounds(n, window):
    half = window // 2
    idx = np.arange(n, dtype=np.int64)
    starts = np.maximum(0, idx - half)
    ends = np.minimum(n, idx + half + 1)
    return starts, ends


def _window_sums(values, starts, ends):
    csum = np.empty(len(values) + 1, dtype=np.float64)
    csum[0] = 0.0
    csum[1:] = np.cumsum(values, dtype=np.float64)
    return csum[ends] - csum[starts]


def _interp_typewell(tvt_query, typewell_tvt, typewell_gr):
    return np.interp(
        tvt_query,
        typewell_tvt,
        typewell_gr,
        left=float(typewell_gr[0]),
        right=float(typewell_gr[-1]),
    )


def _compute_typewell_correlation_for_well(
    well_df,
    typewell_one,
    max_lag=60,
    window=51,
):
    gr_col = _first_existing_column(well_df, ["GR", "gr", "gr_filled"])
    tvt_input_col = "tvt_input_filled" if "tvt_input_filled" in well_df.columns else _first_existing_column(
        well_df, ["TVT_input", "tvt_input", "TVT", "tvt"]
    )
    tw_tvt_col = _first_existing_column(typewell_one, ["TVT", "tvt"])
    tw_gr_col = _first_existing_column(typewell_one, ["GR", "gr"])

    local_gr = _as_numpy(well_df[gr_col]).astype(np.float64)
    prior_tvt = _as_numpy(well_df[tvt_input_col]).astype(np.float64)
    typewell_tvt = _as_numpy(typewell_one[tw_tvt_col]).astype(np.float64)
    typewell_gr = _as_numpy(typewell_one[tw_gr_col]).astype(np.float64)

    order = np.argsort(typewell_tvt, kind="mergesort")
    typewell_tvt = typewell_tvt[order]
    typewell_gr = typewell_gr[order]
    valid_tw = np.isfinite(typewell_tvt) & np.isfinite(typewell_gr)
    typewell_tvt = typewell_tvt[valid_tw]
    typewell_gr = typewell_gr[valid_tw]
    if len(typewell_tvt) < 2:
        raise ValueError("Typewell must contain at least two finite TVT/GR rows")

    gr_median = np.nanmedian(local_gr)
    if not np.isfinite(gr_median):
        gr_median = 0.0
    local_gr = np.where(np.isfinite(local_gr), local_gr, gr_median)

    prior_median = np.nanmedian(prior_tvt)
    if not np.isfinite(prior_median):
        prior_median = float(np.median(typewell_tvt))
    prior_tvt = np.where(np.isfinite(prior_tvt), prior_tvt, prior_median)

    n = len(local_gr)
    starts, ends = _rolling_window_bounds(n, window)
    counts = (ends - starts).astype(np.float64)
    sum_x = _window_sums(local_gr, starts, ends)
    sum_x2 = _window_sums(local_gr * local_gr, starts, ends)
    var_x = np.maximum(sum_x2 - (sum_x * sum_x / counts), 0.0)

    lags = np.arange(-max_lag, max_lag + 1, dtype=np.float64)
    corr_by_lag = np.empty((len(lags), n), dtype=np.float32)

    for lag_idx, lag in enumerate(lags):
        tw_values = _interp_typewell(prior_tvt + lag, typewell_tvt, typewell_gr)
        sum_y = _window_sums(tw_values, starts, ends)
        sum_y2 = _window_sums(tw_values * tw_values, starts, ends)
        sum_xy = _window_sums(local_gr * tw_values, starts, ends)
        cov = sum_xy - (sum_x * sum_y / counts)
        var_y = np.maximum(sum_y2 - (sum_y * sum_y / counts), 0.0)
        denom = np.sqrt(var_x * var_y) + 1e-6
        corr_by_lag[lag_idx] = np.where(counts >= 3, cov / denom, 0.0).astype(np.float32)

    best_idx = np.argmax(corr_by_lag, axis=0)
    row_idx = np.arange(n)
    best_lag = lags[best_idx].astype(np.float32)
    best_corr = corr_by_lag[best_idx, row_idx].astype(np.float32)
    lag0_idx = int(np.where(lags == 0)[0][0])
    corr_at_lag_0 = corr_by_lag[lag0_idx].astype(np.float32)
    corr_mean = corr_by_lag.mean(axis=0)
    corr_median = np.median(corr_by_lag, axis=0)
    corr_std = corr_by_lag.std(axis=0)
    corr_max_minus_mean = (best_corr - corr_mean).astype(np.float32)
    corr_peakiness = ((best_corr - corr_median) / (corr_std + 1e-6)).astype(np.float32)
    typewell_gr_at_best_lag = _interp_typewell(
        prior_tvt + best_lag.astype(np.float64), typewell_tvt, typewell_gr
    ).astype(np.float32)
    typewell_gr_minus_local = (typewell_gr_at_best_lag - local_gr).astype(np.float32)

    return {
        "typewell_best_lag": best_lag,
        "typewell_best_corr": best_corr,
        "typewell_corr_at_lag_0": corr_at_lag_0,
        "typewell_corr_max_minus_mean": corr_max_minus_mean,
        "typewell_corr_peakiness": corr_peakiness,
        "typewell_gr_at_best_lag": typewell_gr_at_best_lag,
        "typewell_gr_minus_local_gr_at_best_lag": typewell_gr_minus_local,
    }


def add_typewell_correlation_features(df, typewell_df, max_lag=60, window=51):
    """
    Add per-well Typewell GR cross-correlation features.

    The search is centered on TVT_input_filled when present, otherwise
    TVT_input/TVT. For every candidate lag in [-max_lag, max_lag], it compares
    the lateral GR curve with the corresponding typewell GR curve using a
    centered rolling Pearson correlation window.
    """
    df = df.copy()
    typewell_df = typewell_df.copy()
    if "well_id" not in df.columns or "well_id" not in typewell_df.columns:
        raise KeyError("Both df and typewell_df must contain well_id")

    feature_names = [
        "typewell_best_lag",
        "typewell_best_corr",
        "typewell_corr_at_lag_0",
        "typewell_corr_max_minus_mean",
        "typewell_corr_peakiness",
        "typewell_gr_at_best_lag",
        "typewell_gr_minus_local_gr_at_best_lag",
    ]
    for name in feature_names:
        df[name] = np.float32(0.0)

    well_values = _as_numpy(df["well_id"])
    unique_wells = list(dict.fromkeys(well_values.tolist()))
    typewell_groups = {well_id: group for well_id, group in typewell_df.groupby("well_id", sort=False)}

    for idx, well_id in enumerate(unique_wells, start=1):
        if well_id not in typewell_groups:
            raise KeyError(f"Missing typewell for well_id={well_id}")
        mask = well_values == well_id
        well_part = df.loc[mask]
        computed = _compute_typewell_correlation_for_well(
            well_part,
            typewell_groups[well_id],
            max_lag=max_lag,
            window=window,
        )
        for name, values in computed.items():
            df.loc[mask, name] = values
        if idx % 50 == 0:
            print(f"Computed typewell correlation features for {idx}/{len(unique_wells)} wells")

    df["typewell_lag_smoothness_rollmean_50"] = df.groupby("well_id")["typewell_best_lag"].transform(
        lambda x: x.rolling(50, center=True, min_periods=1).mean()
    )
    df[
        feature_names + ["typewell_lag_smoothness_rollmean_50"]
    ] = df[
        feature_names + ["typewell_lag_smoothness_rollmean_50"]
    ].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return df


def add_exp004_gr_features(df):
    """
    Expanded GR rolling statistics and per-well normalization for exp004.

    Adds only the features listed in experiments/specs/exp004.json.
    """
    df = df.copy()

    gr_col = "gr" if "gr" in df.columns else "GR" if "GR" in df.columns else None
    md_col = "md" if "md" in df.columns else "MD" if "MD" in df.columns else None
    if gr_col is None or md_col is None or "well_id" not in df.columns:
        print("WARNING: missing gr/MD/well_id columns, skipping exp004 features")
        return df

    grouped_gr = df.groupby("well_id", sort=False)[gr_col]
    for window in (5, 20, 50):
        df[f"gr_rolling_mean_{window}"] = grouped_gr.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    for window in (5, 20, 50, 100):
        df[f"gr_rolling_std_{window}"] = grouped_gr.transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0)
        )

    for window in (30, 100):
        df[f"gr_rolling_min_{window}"] = grouped_gr.transform(
            lambda s: s.rolling(window, min_periods=1).min()
        )
        df[f"gr_rolling_max_{window}"] = grouped_gr.transform(
            lambda s: s.rolling(window, min_periods=1).max()
        )
        df[f"gr_rolling_slope_{window}"] = _rolling_slope_by_group(
            df, gr_col, md_col, "well_id", window
        )

    for period in (1, 3, 5, 20):
        df[f"gr_diff_{period}"] = grouped_gr.transform(
            lambda s: s.diff(period)
        ).fillna(0)

    df["gr_pct_rank_well"] = grouped_gr.transform(lambda s: s.rank(pct=True))
    df["gr_zscore_well"] = grouped_gr.transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-8)
    )

    well_stats = df.groupby("well_id", sort=False)[gr_col].agg(
        ["mean", "std", "min", "max", "skew"]
    )
    well_stats.columns = [
        "gr_well_mean",
        "gr_well_std",
        "gr_well_min",
        "gr_well_max",
        "gr_well_skew",
    ]
    if GPU:
        df = df.merge(well_stats, on="well_id", how="left")
    else:
        df = df.join(well_stats, on="well_id")

    return df


# ============================================================
# MASTER FEATURE BUILDER
# ============================================================

def build_features(df, typewell_df=None, feature_set="baseline"):
    """
    Build features for a given feature set name.

    Args:
        df: Raw well DataFrame
        typewell_df: Typewell reference DataFrame (optional)
        feature_set: 'baseline', 'v1', 'v2', etc.

    Returns:
        DataFrame with engineered features
    """
    # Always start with basics
    df = add_depth_features(df)
    df = add_gr_rolling_features(df)
    df = add_gr_gradient_features(df)
    df = add_gr_normalization_features(df)
    df = add_trajectory_features(df)
    df = add_well_context_features(df)

    if feature_set in ("v1", "v2", "v3", "full"):
        df = add_gr_lag_lead_features(df)

    if feature_set in ("v2", "v3", "full") and typewell_df is not None:
        df = add_typewell_correlation_features(df, typewell_df)

    # Drop rows with too many NaNs
    if GPU:
        import cudf
        df = df.fillna(-999)
    else:
        df = df.fillna(-999)

    return df
