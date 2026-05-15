"""
Post-processing
===============
Apply physical constraints to TVT predictions.
TVT must be smooth and within physically plausible bounds.
"""

import numpy as np


def smooth_predictions(preds: np.ndarray, window: int = 11, polyorder: int = 3):
    """
    Apply Savitzky-Golay smoothing to enforce TVT smoothness.
    TVT cannot change abruptly in real geology.

    Args:
        preds: Raw TVT predictions
        window: Smoothing window (must be odd)
        polyorder: Polynomial order for SG filter

    Returns:
        Smoothed predictions
    """
    try:
        from scipy.signal import savgol_filter
        # Ensure window is odd
        if window % 2 == 0:
            window += 1
        window = min(window, len(preds) - 1 if len(preds) % 2 == 0 else len(preds))
        if window < polyorder + 1:
            return preds
        return savgol_filter(preds, window_length=window, polyorder=polyorder)
    except ImportError:
        # Fallback: simple moving average
        return np.convolve(preds, np.ones(window) / window, mode="same")


def clip_to_physical_bounds(preds: np.ndarray, p_low: float = 1.0, p_high: float = 99.0):
    """
    Clip predictions to percentile bounds from training data.
    Prevents extrapolation beyond geological reality.
    """
    low = np.percentile(preds, p_low)
    high = np.percentile(preds, p_high)
    return np.clip(preds, low, high)


def postprocess_per_well(preds: np.ndarray, well_ids: np.ndarray,
                          smooth: bool = True, clip: bool = True):
    """
    Apply post-processing per well (smoothing must respect well boundaries).

    Args:
        preds: All predictions concatenated
        well_ids: Well ID for each row
        smooth: Apply Savitzky-Golay smoothing
        clip: Apply percentile clipping

    Returns:
        Post-processed predictions
    """
    result = preds.copy()
    unique_wells = np.unique(well_ids)

    for well in unique_wells:
        mask = well_ids == well
        well_preds = preds[mask]

        if smooth and len(well_preds) > 11:
            well_preds = smooth_predictions(well_preds)

        if clip:
            well_preds = clip_to_physical_bounds(well_preds)

        result[mask] = well_preds

    return result
