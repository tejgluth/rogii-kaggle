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

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


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


@njit(cache=True)
def _dtw_distance_band_numba(a, b, start, end, radius):
    n = end - start
    inf = 1.0e30
    prev = np.empty(n + 1, dtype=np.float64)
    curr = np.empty(n + 1, dtype=np.float64)
    for j in range(n + 1):
        prev[j] = inf
        curr[j] = inf
    prev[0] = 0.0

    for i in range(1, n + 1):
        for j in range(n + 1):
            curr[j] = inf

        j_start = i - radius
        if j_start < 1:
            j_start = 1
        j_end = i + radius
        if j_end > n:
            j_end = n

        av = a[start + i - 1]
        for j in range(j_start, j_end + 1):
            diff = av - b[start + j - 1]
            cost = diff * diff
            best_prev = prev[j]
            if curr[j - 1] < best_prev:
                best_prev = curr[j - 1]
            if prev[j - 1] < best_prev:
                best_prev = prev[j - 1]
            curr[j] = cost + best_prev

        tmp = prev
        prev = curr
        curr = tmp

    return np.sqrt(prev[n] / n)


@njit(cache=True)
def _compute_dtw_anchor_features_numba(local_gr, tw_by_lag, sample_pos, lags, half_window, radius):
    n_samples = len(sample_pos)
    n_lags = len(lags)
    out_best_lag = np.empty(n_samples, dtype=np.float32)
    out_min_distance = np.empty(n_samples, dtype=np.float32)
    out_lag0_distance = np.empty(n_samples, dtype=np.float32)
    out_spread = np.empty(n_samples, dtype=np.float32)
    out_confidence = np.empty(n_samples, dtype=np.float32)

    lag0_idx = 0
    for lag_idx in range(n_lags):
        if lags[lag_idx] == 0:
            lag0_idx = lag_idx
            break

    n_rows = len(local_gr)
    for sample_idx in range(n_samples):
        center = sample_pos[sample_idx]
        start = center - half_window
        if start < 0:
            start = 0
        end = center + half_window + 1
        if end > n_rows:
            end = n_rows

        best_distance = 1.0e30
        second_best = 1.0e30
        worst_distance = -1.0
        best_lag = 0.0
        lag0_distance = 0.0

        for lag_idx in range(n_lags):
            distance = _dtw_distance_band_numba(
                local_gr,
                tw_by_lag[lag_idx],
                start,
                end,
                radius,
            )
            if lag_idx == lag0_idx:
                lag0_distance = distance
            if distance < best_distance:
                second_best = best_distance
                best_distance = distance
                best_lag = lags[lag_idx]
            elif distance < second_best:
                second_best = distance
            if distance > worst_distance:
                worst_distance = distance

        out_best_lag[sample_idx] = best_lag
        out_min_distance[sample_idx] = best_distance
        out_lag0_distance[sample_idx] = lag0_distance
        out_spread[sample_idx] = worst_distance - best_distance
        out_confidence[sample_idx] = (second_best - best_distance) / (best_distance + 1.0e-6)

    return out_best_lag, out_min_distance, out_lag0_distance, out_spread, out_confidence


def _compute_dtw_anchor_features_python(local_gr, tw_by_lag, sample_pos, lags, half_window, radius):
    out = [[], [], [], [], []]
    lag0_idx = int(np.where(lags == 0)[0][0]) if np.any(lags == 0) else 0
    for center in sample_pos:
        start = max(0, int(center) - half_window)
        end = min(len(local_gr), int(center) + half_window + 1)
        distances = [
            _dtw_distance_band_numba(local_gr, tw_by_lag[lag_idx], start, end, radius)
            for lag_idx in range(len(lags))
        ]
        distances = np.asarray(distances, dtype=np.float64)
        order = np.argsort(distances)
        best_idx = int(order[0])
        best = float(distances[best_idx])
        second = float(distances[order[1]]) if len(order) > 1 else best
        out[0].append(float(lags[best_idx]))
        out[1].append(best)
        out[2].append(float(distances[lag0_idx]))
        out[3].append(float(distances.max() - best))
        out[4].append(float((second - best) / (best + 1e-6)))
    return tuple(np.asarray(values, dtype=np.float32) for values in out)


def _compute_dtw_lag_features_for_well(
    well_df,
    typewell_one,
    max_lag=40,
    lag_step=2,
    window=41,
    radius=5,
    sample_step=5,
):
    gr_col = "gr_filled" if "gr_filled" in well_df.columns else _first_existing_column(
        well_df, ["GR", "gr"]
    )
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
    local_gr = np.where(np.isfinite(local_gr), local_gr, gr_median).astype(np.float64)

    prior_median = np.nanmedian(prior_tvt)
    if not np.isfinite(prior_median):
        prior_median = float(np.median(typewell_tvt))
    prior_tvt = np.where(np.isfinite(prior_tvt), prior_tvt, prior_median).astype(np.float64)

    n = len(local_gr)
    sample_pos = np.arange(0, n, sample_step, dtype=np.int64)
    if len(sample_pos) == 0 or sample_pos[-1] != n - 1:
        sample_pos = np.r_[sample_pos, n - 1].astype(np.int64)

    lags = np.arange(-max_lag, max_lag + 1, lag_step, dtype=np.float64)
    tw_by_lag = np.empty((len(lags), n), dtype=np.float64)
    for lag_idx, lag in enumerate(lags):
        tw_by_lag[lag_idx] = _interp_typewell(prior_tvt + lag, typewell_tvt, typewell_gr)

    half_window = window // 2
    if NUMBA_AVAILABLE:
        anchors = _compute_dtw_anchor_features_numba(
            local_gr,
            tw_by_lag,
            sample_pos,
            lags,
            half_window,
            radius,
        )
    else:
        anchors = _compute_dtw_anchor_features_python(
            local_gr,
            tw_by_lag,
            sample_pos,
            lags,
            half_window,
            radius,
        )

    row_pos = np.arange(n, dtype=np.float64)
    sample_pos_float = sample_pos.astype(np.float64)
    return {
        "dtw_best_lag": np.interp(row_pos, sample_pos_float, anchors[0]).astype(np.float32),
        "dtw_min_distance": np.interp(row_pos, sample_pos_float, anchors[1]).astype(np.float32),
        "dtw_distance_at_lag_0": np.interp(row_pos, sample_pos_float, anchors[2]).astype(np.float32),
        "dtw_distance_spread": np.interp(row_pos, sample_pos_float, anchors[3]).astype(np.float32),
        "dtw_confidence": np.interp(row_pos, sample_pos_float, anchors[4]).astype(np.float32),
    }


def add_dtw_lag_features(
    df,
    typewell_df,
    max_lag=40,
    lag_step=2,
    window=41,
    radius=5,
    sample_step=5,
):
    """
    Add DTW lag features between a lateral GR window and its well's typewell GR.

    DTW is computed on every sample_step-th row, then linearly interpolated
    back to all rows. The search is centered on TVT_input_filled when present.
    """
    df = df.copy()
    typewell_df = typewell_df.copy()
    if "well_id" not in df.columns or "well_id" not in typewell_df.columns:
        raise KeyError("Both df and typewell_df must contain well_id")

    feature_names = [
        "dtw_best_lag",
        "dtw_min_distance",
        "dtw_distance_at_lag_0",
        "dtw_distance_spread",
        "dtw_confidence",
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
        computed = _compute_dtw_lag_features_for_well(
            df.loc[mask],
            typewell_groups[well_id],
            max_lag=max_lag,
            lag_step=lag_step,
            window=window,
            radius=radius,
            sample_step=sample_step,
        )
        for name, values in computed.items():
            df.loc[mask, name] = values
        if idx % 25 == 0:
            print(f"Computed DTW lag features for {idx}/{len(unique_wells)} wells")

    df["dtw_best_lag_rollmean_50"] = df.groupby("well_id")["dtw_best_lag"].transform(
        lambda x: x.rolling(50, center=True, min_periods=1).mean()
    )
    all_features = feature_names + ["dtw_best_lag_rollmean_50"]
    df[all_features] = df[all_features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return df


def _prepare_typewell_geology_runs(typewell_one):
    tw_tvt_col = _first_existing_column(typewell_one, ["TVT", "tvt"])
    if "Geology" not in typewell_one.columns:
        raise KeyError("typewell_df must contain a Geology column")

    tvt = _as_numpy(typewell_one[tw_tvt_col]).astype(np.float64)
    geology_raw = _as_numpy(typewell_one["Geology"])
    geology = np.asarray(
        [
            "UNKNOWN" if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "" else str(value)
            for value in geology_raw
        ],
        dtype=object,
    )

    order = np.argsort(tvt, kind="mergesort")
    tvt = tvt[order]
    geology = geology[order]
    valid = np.isfinite(tvt)
    tvt = tvt[valid]
    geology = geology[valid]
    if len(tvt) == 0:
        raise ValueError("Typewell must contain at least one finite TVT row")

    run_id = np.zeros(len(tvt), dtype=np.int64)
    run_starts = [0]
    run_labels = [geology[0]]
    for idx in range(1, len(tvt)):
        if geology[idx] != geology[idx - 1]:
            run_starts.append(idx)
            run_labels.append(geology[idx])
        run_id[idx] = len(run_starts) - 1

    run_starts = np.asarray(run_starts, dtype=np.int64)
    run_ends = np.r_[run_starts[1:] - 1, len(tvt) - 1].astype(np.int64)
    run_labels = np.asarray(run_labels, dtype=object)
    run_top = tvt[run_starts]
    run_bottom = tvt[run_ends]

    step = np.nanmedian(np.diff(tvt)) if len(tvt) > 1 else 0.0
    if not np.isfinite(step) or step < 0:
        step = 0.0
    run_thickness = np.maximum(run_bottom - run_top + step, 0.0).astype(np.float32)
    return tvt, geology, run_id, run_labels, run_top, run_bottom, run_thickness


def _nearest_sorted_indices(sorted_values, query_values):
    pos = np.searchsorted(sorted_values, query_values, side="left")
    pos = np.clip(pos, 0, len(sorted_values) - 1)
    prev_pos = np.maximum(pos - 1, 0)
    choose_prev = np.abs(query_values - sorted_values[prev_pos]) < np.abs(query_values - sorted_values[pos])
    return np.where(choose_prev, prev_pos, pos)


def add_typewell_geology_features(df, typewell_df):
    """
    Add exp009 Typewell geology features using existing DTW lag alignment.

    This function does not compute DTW. It expects dtw_best_lag and a TVT prior
    column, then looks up the aligned typewell geology label and containing
    geology-layer geometry for each lateral row.
    """
    df = df.copy()
    typewell_df = typewell_df.copy()
    if "well_id" not in df.columns or "well_id" not in typewell_df.columns:
        raise KeyError("Both df and typewell_df must contain well_id")
    if "dtw_best_lag" not in df.columns:
        raise KeyError("add_typewell_geology_features requires dtw_best_lag")

    tvt_input_col = "tvt_input_filled" if "tvt_input_filled" in df.columns else _first_existing_column(
        df, ["TVT_input", "tvt_input", "TVT", "tvt"]
    )

    categorical_features = [
        "tw_geology_at_aligned_tvt",
        "tw_geology_above_layer",
        "tw_geology_below_layer",
    ]
    numeric_features = [
        "tw_geology_layer_thickness",
        "tw_geology_distance_to_layer_top",
        "tw_geology_distance_to_layer_bottom",
    ]
    for name in categorical_features:
        df[name] = "UNKNOWN"
    for name in numeric_features:
        df[name] = np.float32(0.0)

    if "Geology" not in typewell_df.columns:
        print("WARNING: typewell_df has no Geology column; using UNKNOWN geology features")
        for name in categorical_features:
            df[name] = df[name].fillna("UNKNOWN").astype(str)
        return df

    well_values = _as_numpy(df["well_id"])
    unique_wells = list(dict.fromkeys(well_values.tolist()))
    typewell_groups = {well_id: group for well_id, group in typewell_df.groupby("well_id", sort=False)}

    for idx, well_id in enumerate(unique_wells, start=1):
        if well_id not in typewell_groups:
            raise KeyError(f"Missing typewell for well_id={well_id}")
        mask = well_values == well_id
        well_part = df.loc[mask]
        prior_tvt = _as_numpy(well_part[tvt_input_col]).astype(np.float64)
        lag = _as_numpy(well_part["dtw_best_lag"]).astype(np.float64)

        (
            typewell_tvt,
            typewell_geology,
            typewell_run_id,
            run_labels,
            run_top,
            run_bottom,
            run_thickness,
        ) = _prepare_typewell_geology_runs(typewell_groups[well_id])

        prior_median = np.nanmedian(prior_tvt)
        if not np.isfinite(prior_median):
            prior_median = float(np.median(typewell_tvt))
        aligned_tvt = np.where(np.isfinite(prior_tvt), prior_tvt, prior_median) + np.where(
            np.isfinite(lag), lag, 0.0
        )
        nearest_idx = _nearest_sorted_indices(typewell_tvt, aligned_tvt)
        row_run = typewell_run_id[nearest_idx]
        label = typewell_geology[nearest_idx]
        above_run = np.maximum(row_run - 1, 0)
        below_run = np.minimum(row_run + 1, len(run_labels) - 1)

        top = run_top[row_run]
        bottom = run_bottom[row_run]
        df.loc[mask, "tw_geology_at_aligned_tvt"] = label
        df.loc[mask, "tw_geology_above_layer"] = run_labels[above_run]
        df.loc[mask, "tw_geology_below_layer"] = run_labels[below_run]
        df.loc[mask, "tw_geology_layer_thickness"] = run_thickness[row_run]
        df.loc[mask, "tw_geology_distance_to_layer_top"] = np.maximum(aligned_tvt - top, 0.0).astype(np.float32)
        df.loc[mask, "tw_geology_distance_to_layer_bottom"] = np.maximum(bottom - aligned_tvt, 0.0).astype(np.float32)

        if idx % 25 == 0:
            print(f"Computed typewell geology features for {idx}/{len(unique_wells)} wells")

    df[numeric_features] = (
        df[numeric_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
    for name in categorical_features:
        df[name] = df[name].fillna("UNKNOWN").astype(str)
    return df


def add_exp011_typewell_geology_context_features(df, typewell_df):
    """
    Add exp011 geology context features around the DTW-aligned typewell TVT.

    This extends the exp009 typewell geology alignment without changing it.
    It expects dtw_best_lag and a TVT prior column, then derives contiguous
    layer-run indexes, offset geology labels, within-layer position, typewell
    GR statistics by aligned geology label, and local GR residuals.
    """
    df = df.copy()
    typewell_df = typewell_df.copy()
    if "well_id" not in df.columns or "well_id" not in typewell_df.columns:
        raise KeyError("Both df and typewell_df must contain well_id")
    if "dtw_best_lag" not in df.columns:
        raise KeyError("add_exp011_typewell_geology_context_features requires dtw_best_lag")
    tvt_input_col = "tvt_input_filled" if "tvt_input_filled" in df.columns else _first_existing_column(
        df, ["TVT_input", "tvt_input", "TVT", "tvt"]
    )
    gr_col = "gr_filled" if "gr_filled" in df.columns else _first_existing_column(
        df, ["GR", "gr"]
    )

    categorical_features = [
        "tw_geology_lag_+5",
        "tw_geology_lag_-5",
        "tw_geology_lag_+15",
        "tw_geology_lag_-15",
    ]
    numeric_features = [
        "tw_geology_layer_index",
        "tw_geology_layer_index_from_bottom",
        "tw_geology_n_layers_total",
        "tw_layer_position_fraction",
        "tw_layer_top_tvt",
        "tw_layer_bottom_tvt",
        "tw_geology_layer_mean_gr",
        "tw_geology_layer_std_gr",
        "gr_minus_layer_mean_gr",
        "gr_minus_layer_mean_gr_zscore",
        "tw_distinct_geologies_in_window_50",
    ]
    for name in categorical_features:
        df[name] = "UNKNOWN"
    for name in numeric_features:
        df[name] = np.float32(0.0)

    if "Geology" not in typewell_df.columns:
        print("WARNING: typewell_df has no Geology column; using neutral exp011 geology context features")
        for name in categorical_features:
            df[name] = df[name].fillna("UNKNOWN").astype(str)
        df[numeric_features] = df[numeric_features].astype("float32")
        return df

    well_values = _as_numpy(df["well_id"])
    unique_wells = list(dict.fromkeys(well_values.tolist()))
    typewell_groups = {well_id: group for well_id, group in typewell_df.groupby("well_id", sort=False)}

    for idx, well_id in enumerate(unique_wells, start=1):
        if well_id not in typewell_groups:
            raise KeyError(f"Missing typewell for well_id={well_id}")
        mask = well_values == well_id
        well_part = df.loc[mask]
        prior_tvt = _as_numpy(well_part[tvt_input_col]).astype(np.float64)
        lag = _as_numpy(well_part["dtw_best_lag"]).astype(np.float64)
        local_gr = _as_numpy(well_part[gr_col]).astype(np.float64)

        typewell_one = typewell_groups[well_id]
        tw_gr_col = _first_existing_column(typewell_one, ["GR", "gr"])
        (
            typewell_tvt,
            typewell_geology,
            typewell_run_id,
            run_labels,
            run_top,
            run_bottom,
            _run_thickness,
        ) = _prepare_typewell_geology_runs(typewell_one)

        tw_tvt_col = _first_existing_column(typewell_one, ["TVT", "tvt"])
        tw_tvt_raw = _as_numpy(typewell_one[tw_tvt_col]).astype(np.float64)
        tw_gr_raw = _as_numpy(typewell_one[tw_gr_col]).astype(np.float64)
        order = np.argsort(tw_tvt_raw, kind="mergesort")
        tw_gr_sorted = tw_gr_raw[order]
        tw_gr_sorted = tw_gr_sorted[np.isfinite(tw_tvt_raw[order])]
        if len(tw_gr_sorted) != len(typewell_tvt):
            tw_gr_sorted = np.interp(
                typewell_tvt,
                tw_tvt_raw[np.isfinite(tw_tvt_raw)],
                tw_gr_raw[np.isfinite(tw_tvt_raw)],
            )

        prior_median = np.nanmedian(prior_tvt)
        if not np.isfinite(prior_median):
            prior_median = float(np.median(typewell_tvt))
        aligned_tvt = np.where(np.isfinite(prior_tvt), prior_tvt, prior_median) + np.where(
            np.isfinite(lag), lag, 0.0
        )
        nearest_idx = _nearest_sorted_indices(typewell_tvt, aligned_tvt)
        row_run = typewell_run_id[nearest_idx]
        row_label = typewell_geology[nearest_idx]

        n_layers = len(run_labels)
        layer_index = row_run + 1
        layer_index_from_bottom = n_layers - row_run
        top = run_top[row_run]
        bottom = run_bottom[row_run]

        for offset in (5, -5, 15, -15):
            offset_idx = _nearest_sorted_indices(typewell_tvt, aligned_tvt + offset)
            sign = "+" if offset > 0 else "-"
            df.loc[mask, f"tw_geology_lag_{sign}{abs(offset)}"] = typewell_geology[offset_idx]

        label_mean: dict[str, float] = {}
        label_std: dict[str, float] = {}
        for label in np.unique(typewell_geology):
            label_mask = typewell_geology == label
            gr_values = tw_gr_sorted[label_mask]
            gr_values = gr_values[np.isfinite(gr_values)]
            if len(gr_values) == 0:
                label_mean[str(label)] = 0.0
                label_std[str(label)] = 1.0
            else:
                label_mean[str(label)] = float(np.mean(gr_values))
                std = float(np.std(gr_values))
                label_std[str(label)] = std if np.isfinite(std) and std > 1e-6 else 1.0

        layer_mean_gr = np.asarray([label_mean[str(label)] for label in row_label], dtype=np.float32)
        layer_std_gr = np.asarray([label_std[str(label)] for label in row_label], dtype=np.float32)
        local_gr_median = np.nanmedian(local_gr)
        if not np.isfinite(local_gr_median):
            local_gr_median = 0.0
        local_gr = np.where(np.isfinite(local_gr), local_gr, local_gr_median)
        gr_residual = (local_gr.astype(np.float32) - layer_mean_gr).astype(np.float32)

        starts = np.searchsorted(typewell_tvt, aligned_tvt - 25.0, side="left")
        ends = np.searchsorted(typewell_tvt, aligned_tvt + 25.0, side="right")
        distinct_counts = np.fromiter(
            (len(set(typewell_geology[start:end])) for start, end in zip(starts, ends)),
            dtype=np.float32,
            count=len(starts),
        )

        df.loc[mask, "tw_geology_layer_index"] = layer_index.astype(np.float32)
        df.loc[mask, "tw_geology_layer_index_from_bottom"] = layer_index_from_bottom.astype(np.float32)
        df.loc[mask, "tw_geology_n_layers_total"] = np.float32(n_layers)
        df.loc[mask, "tw_layer_position_fraction"] = (
            (aligned_tvt - top) / (bottom - top + 1e-6)
        ).astype(np.float32)
        df.loc[mask, "tw_layer_top_tvt"] = top.astype(np.float32)
        df.loc[mask, "tw_layer_bottom_tvt"] = bottom.astype(np.float32)
        df.loc[mask, "tw_geology_layer_mean_gr"] = layer_mean_gr
        df.loc[mask, "tw_geology_layer_std_gr"] = layer_std_gr
        df.loc[mask, "gr_minus_layer_mean_gr"] = gr_residual
        df.loc[mask, "gr_minus_layer_mean_gr_zscore"] = (gr_residual / (layer_std_gr + 1e-6)).astype(np.float32)
        df.loc[mask, "tw_distinct_geologies_in_window_50"] = distinct_counts

        if idx % 25 == 0:
            print(f"Computed exp011 geology context features for {idx}/{len(unique_wells)} wells")

    df[numeric_features] = (
        df[numeric_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
    for name in categorical_features:
        df[name] = df[name].fillna("UNKNOWN").astype(str)
    return df


def add_fold_safe_target_encoding(
    train_df,
    test_df,
    target_col,
    category_col,
    output_col,
    folds,
    smoothing=20.0,
):
    """
    Add fold-safe target encoding for a categorical column.

    ``folds`` must be an iterable of (train_idx, val_idx) arrays. Validation
    rows are encoded only from their training-fold labels; test rows are encoded
    from the full training data.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    global_mean = float(train_df[target_col].mean())
    train_encoded = np.full(len(train_df), global_mean, dtype=np.float32)

    def build_mapping(frame):
        stats = frame.groupby(category_col, sort=False)[target_col].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        return smooth.to_dict()

    for tr_idx, val_idx in folds:
        mapping = build_mapping(train_df.iloc[tr_idx])
        encoded = train_df.iloc[val_idx][category_col].map(mapping).fillna(global_mean)
        train_encoded[val_idx] = encoded.to_numpy(dtype=np.float32)

    full_mapping = build_mapping(train_df)
    train_df[output_col] = train_encoded
    test_df[output_col] = test_df[category_col].map(full_mapping).fillna(global_mean).astype("float32")
    return train_df, test_df


def add_alignment_interaction_features(df):
    """
    Add interaction features between Typewell cross-correlation and DTW lag estimates.

    Requires add_typewell_correlation_features and add_dtw_lag_features to have
    already populated their respective lag/confidence columns.
    """
    df = df.copy()
    required = [
        "typewell_best_lag",
        "typewell_best_corr",
        "dtw_best_lag",
        "dtw_confidence",
    ]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for alignment interaction features: {missing}")

    df["xcorr_minus_dtw_lag"] = df["typewell_best_lag"] - df["dtw_best_lag"]
    df["lag_agreement_strength"] = df["typewell_best_corr"] * df["dtw_confidence"]
    interaction_features = ["xcorr_minus_dtw_lag", "lag_agreement_strength"]
    df[interaction_features] = (
        df[interaction_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
    return df


def _robust_zscore(values):
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values)
    if not np.isfinite(median):
        median = 0.0
    mad = np.nanmedian(np.abs(values - median))
    scale = mad * 1.4826
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    filled = np.where(np.isfinite(values), values, median)
    return ((filled - median) / scale).astype(np.float64)


def _compute_exp008_xcorr_for_well(
    well_df,
    typewell_one,
    max_lag=80,
    windows=(21, 51, 101),
):
    gr_col = "gr_filled" if "gr_filled" in well_df.columns else _first_existing_column(
        well_df, ["GR", "gr"]
    )
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

    prior_median = np.nanmedian(prior_tvt)
    if not np.isfinite(prior_median):
        prior_median = float(np.median(typewell_tvt))
    prior_tvt = np.where(np.isfinite(prior_tvt), prior_tvt, prior_median).astype(np.float64)

    local_norm = _robust_zscore(local_gr)
    typewell_norm = _robust_zscore(typewell_gr)

    n = len(local_norm)
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.float64)
    results = {}
    best_lags = []

    for window in windows:
        starts, ends = _rolling_window_bounds(n, window)
        counts = (ends - starts).astype(np.float64)
        sum_x = _window_sums(local_norm, starts, ends)
        sum_x2 = _window_sums(local_norm * local_norm, starts, ends)
        var_x = np.maximum(sum_x2 - (sum_x * sum_x / counts), 0.0)
        corr_by_lag = np.empty((len(lags), n), dtype=np.float32)

        for lag_idx, lag in enumerate(lags):
            tw_values = _interp_typewell(
                prior_tvt + lag,
                typewell_tvt,
                typewell_norm,
            )
            sum_y = _window_sums(tw_values, starts, ends)
            sum_y2 = _window_sums(tw_values * tw_values, starts, ends)
            sum_xy = _window_sums(local_norm * tw_values, starts, ends)
            cov = sum_xy - (sum_x * sum_y / counts)
            var_y = np.maximum(sum_y2 - (sum_y * sum_y / counts), 0.0)
            denom = np.sqrt(var_x * var_y)
            corr_by_lag[lag_idx] = np.divide(
                cov,
                denom,
                out=np.zeros(n, dtype=np.float64),
                where=(counts >= 3) & (denom > 1e-8),
            ).astype(np.float32)

        best_idx = np.argmax(corr_by_lag, axis=0)
        row_idx = np.arange(n)
        best_lag = lags[best_idx].astype(np.float32)
        best_corr = corr_by_lag[best_idx, row_idx].astype(np.float32)
        results[f"tw_xcorr_best_lag_w{window}"] = best_lag
        results[f"tw_xcorr_best_corr_w{window}"] = best_corr
        best_lags.append(best_lag.astype(np.float64))

    lag_stack = np.vstack(best_lags)
    median_lag = np.median(lag_stack, axis=0)
    results["tw_xcorr_lag_agreement"] = np.std(lag_stack, axis=0).astype(np.float32)
    consensus_starts, consensus_ends = _rolling_window_bounds(n, 50)
    consensus_counts = (consensus_ends - consensus_starts).astype(np.float64)
    consensus = (
        _window_sums(median_lag, consensus_starts, consensus_ends) / consensus_counts
    ).astype(np.float32)
    results["tw_xcorr_lag_consensus_smooth_50"] = consensus

    typewell_at_best = _interp_typewell(
        prior_tvt + median_lag,
        typewell_tvt,
        typewell_norm,
    ).astype(np.float32)
    results["tw_norm_gr_residual_at_best_lag"] = (
        local_norm.astype(np.float32) - typewell_at_best
    ).astype(np.float32)
    results["tw_typewell_gr_zscore_at_best_lag"] = typewell_at_best
    return results


def add_exp008_typewell_xcorr_features(
    df,
    typewell_df,
    max_lag=80,
    windows=(21, 51, 101),
):
    """
    Add exp008 normalized multi-scale Typewell alignment features.

    Lateral GR and typewell GR are robust z-scored per well/typewell before
    centered rolling Pearson correlation is computed over candidate TVT lags.
    """
    df = df.copy()
    typewell_df = typewell_df.copy()
    if "well_id" not in df.columns or "well_id" not in typewell_df.columns:
        raise KeyError("Both df and typewell_df must contain well_id")

    feature_names = [
        "tw_xcorr_best_lag_w21",
        "tw_xcorr_best_corr_w21",
        "tw_xcorr_best_lag_w51",
        "tw_xcorr_best_corr_w51",
        "tw_xcorr_best_lag_w101",
        "tw_xcorr_best_corr_w101",
        "tw_xcorr_lag_agreement",
        "tw_xcorr_lag_consensus_smooth_50",
        "tw_norm_gr_residual_at_best_lag",
        "tw_typewell_gr_zscore_at_best_lag",
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
        computed = _compute_exp008_xcorr_for_well(
            df.loc[mask],
            typewell_groups[well_id],
            max_lag=max_lag,
            windows=windows,
        )
        for name, values in computed.items():
            df.loc[mask, name] = values
        if idx % 25 == 0:
            print(f"Computed exp008 xcorr features for {idx}/{len(unique_wells)} wells")

    df[feature_names] = (
        df[feature_names]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
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


def add_exp005_trajectory_features(df):
    """
    Trajectory-derived geometry features for exp005.

    Adds only the features listed in experiments/specs/exp005.json.
    """
    df = df.copy()

    required = {"well_id", "MD", "X", "Y", "Z"}
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"WARNING: missing columns for exp005 trajectory features: {missing}")
        return df

    grouped = df.groupby("well_id", sort=False)

    md_step = grouped["MD"].diff().fillna(0)
    x_step = grouped["X"].diff().fillna(0)
    y_step = grouped["Y"].diff().fillna(0)
    z_step = grouped["Z"].diff().fillna(0)
    horizontal_step = np.sqrt(x_step**2 + y_step**2)

    df["md_step"] = md_step
    df["x_diff_5"] = grouped["X"].diff(5).fillna(0)
    df["y_diff_5"] = grouped["Y"].diff(5).fillna(0)
    df["z_diff_5"] = grouped["Z"].diff(5).fillna(0)

    safe_md_step = md_step.abs().replace(0, np.nan)
    z_gradient = (z_step / safe_md_step).fillna(0)
    df["z_gradient_rollmean_30"] = z_gradient.groupby(df["well_id"], sort=False).transform(
        lambda s: s.rolling(30, min_periods=1).mean()
    )

    xyz_step = np.sqrt(x_step**2 + y_step**2 + z_step**2)
    df["dls_xyz"] = (xyz_step / safe_md_step).fillna(0)
    df["dls_xyz_rollmean_30"] = df.groupby("well_id", sort=False)["dls_xyz"].transform(
        lambda s: s.rolling(30, min_periods=1).mean()
    )

    df["lateral_dist_from_heel"] = horizontal_step.groupby(df["well_id"], sort=False).cumsum()
    x_from_heel = grouped["X"].transform(lambda s: s - s.iloc[0])
    y_from_heel = grouped["Y"].transform(lambda s: s - s.iloc[0])
    df["xy_dist_from_heel"] = np.sqrt(x_from_heel**2 + y_from_heel**2)

    z_smooth = grouped["Z"].transform(
        lambda s: s.rolling(30, center=True, min_periods=1).mean()
    )
    z_curvature = z_smooth.groupby(df["well_id"], sort=False).diff().groupby(
        df["well_id"], sort=False
    ).diff()
    df["z_curvature"] = z_curvature.fillna(0)

    df["inclination_proxy"] = np.arctan2(horizontal_step, z_step).fillna(0)
    df["azimuth_proxy"] = np.arctan2(y_step, x_step).fillna(0)

    heel_z = grouped["Z"].transform("first")
    df["depth_below_heel_z"] = heel_z - df["Z"]

    exp005_features = [
        "x_diff_5",
        "y_diff_5",
        "z_diff_5",
        "z_gradient_rollmean_30",
        "dls_xyz",
        "dls_xyz_rollmean_30",
        "lateral_dist_from_heel",
        "xy_dist_from_heel",
        "z_curvature",
        "md_step",
        "inclination_proxy",
        "azimuth_proxy",
        "depth_below_heel_z",
    ]
    df[exp005_features] = (
        df[exp005_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )

    return df


# ============================================================
# MASTER FEATURE BUILDER
# ============================================================

EXP006_FEATURES = [
    "MD",
    "X",
    "Y",
    "Z",
    "depth_from_heel",
    "depth_fraction",
    "gr_filled",
    "gr_rolling_mean_10",
    "gr_rolling_mean_30",
    "gr_rolling_std_10",
    "gr_rolling_mean_100",
    "tvt_input_filled",
    "x_diff",
    "y_diff",
    "z_diff",
    "lateral_speed",
    "dtw_best_lag",
    "dtw_min_distance",
    "dtw_distance_at_lag_0",
    "dtw_distance_spread",
    "dtw_best_lag_rollmean_50",
    "dtw_confidence",
]

EXP009_DTW_FEATURES = [
    "dtw_best_lag",
    "dtw_min_distance",
    "dtw_best_lag_rollmean_50",
    "dtw_confidence",
]

EXP009_GEOLOGY_CATEGORICAL_FEATURES = [
    "tw_geology_at_aligned_tvt",
    "tw_geology_above_layer",
    "tw_geology_below_layer",
]

EXP009_GEOLOGY_NUMERIC_FEATURES = [
    "tw_geology_target_enc",
    "tw_geology_layer_thickness",
    "tw_geology_distance_to_layer_top",
    "tw_geology_distance_to_layer_bottom",
]

EXP009_FEATURES = [
    "MD",
    "X",
    "Y",
    "Z",
    "depth_from_heel",
    "depth_fraction",
    "gr_filled",
    "gr_rolling_mean_10",
    "gr_rolling_mean_30",
    "gr_rolling_std_10",
    "gr_rolling_mean_100",
    "tvt_input_filled",
    "x_diff",
    "y_diff",
    "z_diff",
    "lateral_speed",
] + EXP009_DTW_FEATURES + EXP009_GEOLOGY_CATEGORICAL_FEATURES + EXP009_GEOLOGY_NUMERIC_FEATURES

EXP011_DEEP_GEOLOGY_CATEGORICAL_FEATURES = [
    "tw_geology_lag_+5",
    "tw_geology_lag_-5",
    "tw_geology_lag_+15",
    "tw_geology_lag_-15",
]

EXP011_GEOLOGY_CATEGORICAL_FEATURES = (
    EXP009_GEOLOGY_CATEGORICAL_FEATURES + EXP011_DEEP_GEOLOGY_CATEGORICAL_FEATURES
)

EXP011_DEEP_GEOLOGY_NUMERIC_FEATURES = [
    "tw_geology_layer_index",
    "tw_geology_layer_index_from_bottom",
    "tw_geology_n_layers_total",
    "tw_layer_position_fraction",
    "tw_layer_top_tvt",
    "tw_layer_bottom_tvt",
    "tw_geology_layer_mean_gr",
    "tw_geology_layer_std_gr",
    "gr_minus_layer_mean_gr",
    "gr_minus_layer_mean_gr_zscore",
    "tw_distinct_geologies_in_window_50",
]

EXP011_GEOLOGY_NUMERIC_FEATURES = EXP009_GEOLOGY_NUMERIC_FEATURES + EXP011_DEEP_GEOLOGY_NUMERIC_FEATURES

EXP011_FEATURES = [
    "MD",
    "X",
    "Y",
    "Z",
    "depth_from_heel",
    "depth_fraction",
    "gr_filled",
    "gr_rolling_mean_10",
    "gr_rolling_mean_30",
    "gr_rolling_std_10",
    "gr_rolling_mean_100",
    "tvt_input_filled",
    "x_diff",
    "y_diff",
    "z_diff",
    "lateral_speed",
] + EXP009_DTW_FEATURES + EXP011_GEOLOGY_CATEGORICAL_FEATURES + EXP011_GEOLOGY_NUMERIC_FEATURES


def add_exp006_baseline_features(
    df,
    global_gr_median=None,
    global_tvt_input_median=None,
):
    """Add the exact baseline feature columns used by exp006."""
    df = df.sort_values(["well_id", "MD"], kind="mergesort").reset_index(drop=True)

    for column in ["MD", "X", "Y", "Z", "GR", "TVT_input"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if global_gr_median is None:
        global_gr_median = float(df["GR"].median())
    if global_tvt_input_median is None:
        global_tvt_input_median = float(df["TVT_input"].median())

    if np.isnan(global_gr_median):
        global_gr_median = 0.0
    if np.isnan(global_tvt_input_median):
        global_tvt_input_median = 0.0

    grouped = df.groupby("well_id", sort=False, group_keys=False)

    md_min = grouped["MD"].transform("min")
    md_max = grouped["MD"].transform("max")
    depth_range = (md_max - md_min).replace(0, np.nan)
    df["depth_from_heel"] = df["MD"] - md_min
    df["depth_fraction"] = (df["depth_from_heel"] / depth_range).fillna(0)

    gr_median = grouped["GR"].transform("median").fillna(global_gr_median)
    df["gr_filled"] = df["GR"].fillna(gr_median).fillna(global_gr_median)
    gr_grouped = df.groupby("well_id", sort=False)["gr_filled"]
    df["gr_rolling_mean_10"] = gr_grouped.transform(
        lambda s: s.rolling(10, center=True, min_periods=1).mean()
    )
    df["gr_rolling_mean_30"] = gr_grouped.transform(
        lambda s: s.rolling(30, center=True, min_periods=1).mean()
    )
    df["gr_rolling_std_10"] = gr_grouped.transform(
        lambda s: s.rolling(10, center=True, min_periods=1).std().fillna(0)
    )
    df["gr_rolling_mean_100"] = gr_grouped.transform(
        lambda s: s.rolling(100, center=True, min_periods=1).mean()
    )

    tvt_input_median = grouped["TVT_input"].transform("median").fillna(global_tvt_input_median)
    df["tvt_input_filled"] = df["TVT_input"].fillna(tvt_input_median).fillna(global_tvt_input_median)

    df["x_diff"] = grouped["X"].diff().fillna(0)
    df["y_diff"] = grouped["Y"].diff().fillna(0)
    df["z_diff"] = grouped["Z"].diff().fillna(0)
    df["lateral_speed"] = np.sqrt(df["x_diff"] ** 2 + df["y_diff"] ** 2 + df["z_diff"] ** 2)
    return df, global_gr_median, global_tvt_input_median


def build_exp006_feature_matrix(
    df,
    typewell_df,
    global_gr_median=None,
    global_tvt_input_median=None,
):
    """Build the exact exp006 DTW feature matrix for reuse by later models."""
    df, global_gr_median, global_tvt_input_median = add_exp006_baseline_features(
        df,
        global_gr_median,
        global_tvt_input_median,
    )
    df = add_dtw_lag_features(
        df,
        typewell_df,
        max_lag=40,
        lag_step=2,
        window=41,
        radius=5,
        sample_step=5,
    )
    df[EXP006_FEATURES] = (
        df[EXP006_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
    return df, global_gr_median, global_tvt_input_median


def build_exp009_feature_matrix(
    df,
    typewell_df,
    global_gr_median=None,
    global_tvt_input_median=None,
):
    """
    Build the exact exp009 DTW + Typewell geology feature matrix.

    The fold-safe ``tw_geology_target_enc`` column is intentionally left for the
    training script to populate after GroupKFold splits are known.
    """
    df, global_gr_median, global_tvt_input_median = add_exp006_baseline_features(
        df,
        global_gr_median,
        global_tvt_input_median,
    )
    df = add_dtw_lag_features(
        df,
        typewell_df,
        max_lag=40,
        lag_step=2,
        window=41,
        radius=5,
        sample_step=5,
    )
    df = add_typewell_geology_features(df, typewell_df)

    numeric_features = [
        feature
        for feature in EXP009_FEATURES
        if feature not in EXP009_GEOLOGY_CATEGORICAL_FEATURES
        and feature != "tw_geology_target_enc"
    ]
    df[numeric_features] = (
        df[numeric_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
    for column in EXP009_GEOLOGY_CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("UNKNOWN").astype(str)
    return df, global_gr_median, global_tvt_input_median


def build_exp011_feature_matrix(
    df,
    typewell_df,
    global_gr_median=None,
    global_tvt_input_median=None,
):
    """
    Build the exp011 deeper Typewell geology feature matrix.

    This extends exp009 with offset geology labels, layer indexes, within-layer
    position, layer GR summaries, GR residuals, and local geology diversity.
    The fold-safe ``tw_geology_target_enc`` column is left for callers that use
    it; CatBoost experiments can drop it and pass geology labels natively.
    """
    df, global_gr_median, global_tvt_input_median = build_exp009_feature_matrix(
        df,
        typewell_df,
        global_gr_median,
        global_tvt_input_median,
    )
    df = add_exp011_typewell_geology_context_features(df, typewell_df)

    numeric_features = [
        feature
        for feature in EXP011_FEATURES
        if feature not in EXP011_GEOLOGY_CATEGORICAL_FEATURES
        and feature != "tw_geology_target_enc"
    ]
    df[numeric_features] = (
        df[numeric_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
    for column in EXP011_GEOLOGY_CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("UNKNOWN").astype(str)
    return df, global_gr_median, global_tvt_input_median


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

    if feature_set in ("exp006", "full") and typewell_df is not None:
        df = add_dtw_lag_features(df, typewell_df)

    if feature_set in ("exp007", "full") and typewell_df is not None:
        df = add_typewell_correlation_features(df, typewell_df)
        df = add_dtw_lag_features(df, typewell_df)
        df = add_alignment_interaction_features(df)

    if feature_set in ("exp008", "full") and typewell_df is not None:
        df = add_exp008_typewell_xcorr_features(df, typewell_df)

    # Drop rows with too many NaNs
    if GPU:
        import cudf
        df = df.fillna(-999)
    else:
        df = df.fillna(-999)

    return df
