from __future__ import annotations
import mne
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import chi2, norm
from typing import Tuple, Optional
import gc
import zlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

def create_mne_raw_from_data(data, channel_names, sampling_frequency, eeg_channel_count=16, ch_types=None):
    """
    Create an MNE Raw object from numpy array data.

    Parameters:
    - data: numpy array of shape (num_time_points, num_channels)
    - channel_names: list of channel names
    - sampling_frequency: sampling frequency of the data
    - eeg_channel_count: number of EEG channels in the data
    - ch_types: list of channel types. If None, defaults to 'eeg' for the first eeg_channel_count channels
      and 'misc' for the rest.

    Returns:
    - raw: MNE Raw object
    """
    if ch_types is None:
        # Default type assignment if `ch_types` is not provided
        ch_types = ['eeg'] * eeg_channel_count + ['misc'] * (data.shape[1] - eeg_channel_count)

    # Ensure that the number of channel names and types match the data shape
    assert len(channel_names) == len(ch_types) == data.shape[1], "Channel names, types, and data dimensions must match."

    # Create an MNE Info object with the properties of your data
    info = mne.create_info(
        ch_names=channel_names, 
        sfreq=sampling_frequency, 
        ch_types=ch_types
    )

    # Transpose data to the shape `num_channels x num_samples`
    raw_data_transposed = data.T

    # Create the Raw object
    raw = mne.io.RawArray(raw_data_transposed, info, verbose=False)

    return raw

def get_eeg_channel_indices(ch_names, channel_groups):
    """
    Obtain indices of EEG channels that need to be considered for analysis,
    omitting auxiliary channels.
    
    Parameters:
        ch_names (list): Full list of channel names, including both EEG and auxiliary channels.
        channel_groups (dict): Dictionary where keys are EEG channel group names (e.g., 'frontal', 'central')
                               and values are lists of EEG channel names in those groups.
    
    Returns:
        Tuple: (List of indices, List of channel names) corresponding to EEG channels to be considered for analysis.
    """

    # Flatten the channel names from the channel_groups dictionary into a set for quick lookup.
    eeg_channel_set = set([ch for group in channel_groups.values() for ch in group])

    # Look up each channel name in the eeg_channel_set to determine if it's EEG.
    # Store the index if the channel is an EEG channel.
    eeg_channel_indices = [i for i, ch in enumerate(ch_names) if ch in eeg_channel_set]
    eeg_channel_names = [ch for ch in ch_names if ch in eeg_channel_set]

    return eeg_channel_indices, eeg_channel_names

def select_channels_and_adjust_data(epoch, select_channels, ch_names, ch_groups):
    if select_channels is not None:
        if isinstance(select_channels[0], str):
            select_channel_names = select_channels
            channel_indices = [ch_names.index(name) for name in select_channel_names]
            epoch = epoch[:, channel_indices]
        elif isinstance(select_channels[0], int):
            select_channel_names = [ch_names[i] for i in select_channels]
            epoch = epoch[:, select_channels]
        # select_channel_types = ['eeg'] * len(select_channels)     
    else:
        all_eeg_channels = []
        for group, channels in ch_groups.items():
            if group in ['prefrontal', 'frontal', 'central', 'temporal', 'parietal', 'occipital']:
                all_eeg_channels.extend(channels)
        channel_indices = [ch_names.index(name) for name in all_eeg_channels]
        epoch = epoch[:, channel_indices]
        select_channel_names = all_eeg_channels
    
    select_channel_types = ['eeg'] * len(select_channel_names)

        # select_channel_names = ch_names
        # select_channel_types = None

    return select_channel_names, select_channel_types, epoch


def calculate_z_score_eeg(eeg, duration=None, sampling_rate=250, peak_threshold=8):
    """
    Calculates the z-score in consecutive bouts of EEG data with specified duration and concatenates them together.
    Removes data with peak z_score above the specified threshold by embedding them with zero before recalculating z_score.
    
    Parameters:
        eeg (numpy array): 2D array of EEG data (time points x channels)
        duration (int): Duration of each segment in seconds
        sampling_rate (int): Sampling rate of EEG data in Hz
        peak_threshold (float): Threshold for z-score peaks to be removed

    Returns:
        numpy array: Z-scored EEG data concatenated across all segments
    """
    if duration is None:
        duration = int(eeg.shape[0]/sampling_rate)

    n_samples = eeg.shape[0]
    n_channels = eeg.shape[1]

    # Calculate the number of samples per segment
    segment_samples = duration * sampling_rate
    n_segments = n_samples // segment_samples

    # Initialize the array to hold the z-scored data
    z_scored_epoch = np.zeros_like(eeg)

    for channel in range(n_channels):
        for segment in range(n_segments):
            start = segment * segment_samples
            end = start + segment_samples

            if end > n_samples:
                break

            segment_eeg = eeg[start:end, channel]

            # Calculate initial z-score for the segment
            initial_z_score = (segment_eeg - np.mean(segment_eeg)) / np.std(segment_eeg)

            # Find peaks where z-score exceeds the threshold
            peaks, _ = find_peaks(np.abs(initial_z_score), height=peak_threshold)

            # Identify zero crossings
            zero_crossings = np.where(np.diff(np.sign(segment_eeg)))[0]

            # Remove data between zero crossings before and after the peak
            for peak in peaks:
                before_zero_crossing = zero_crossings[zero_crossings < peak]
                after_zero_crossing = zero_crossings[zero_crossings > peak]
                
                if len(before_zero_crossing) > 0 and len(after_zero_crossing) > 0:
                    start_idx = before_zero_crossing[-1] + 1
                    end_idx = after_zero_crossing[0] + 1
                    segment_eeg[start_idx:end_idx] = 0

            # Recalculate the z-score after removing peaks
            segment_mean = np.mean(segment_eeg)
            segment_std = np.std(segment_eeg)
            
            if segment_std == 0:  # Avoid division by zero
                z_scored_segment = np.zeros_like(segment_eeg)
            else:
                z_scored_segment = (segment_eeg - segment_mean) / segment_std

            z_scored_epoch[start:end, channel] = z_scored_segment

    return z_scored_epoch

def remove_outliers(data, factor=1.5):
    """
    Remove outliers from a dataset using the interquartile range (IQR) method.
    """
    
    data = np.array(data)
    # data = data[~np.isnan(data)]

    q1, q3 = np.nanpercentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    mask = (data >= lower_bound) & (data <= upper_bound)
    filtered_data = data[mask]
    # indices = np.where(mask)[0]  # Get the indices of accepted data samples
    
    return filtered_data, mask

def bottom_left_off_diagonal(array):
    """Extracts off-diagonal elements from the bottom-left rectangle of a matrix."""

    rows, cols = array.shape
    result = []

    for i in range(1, rows):
        for j in range(0, i):
            result.append(array[i, j])
    return np.array(result)

def detect_and_interpolate_outliers(time_series, iqr_factor=1.5):
    """
    Detects outliers in a time series using the IQR method and interpolates them using linear interpolation,
    while preserving the original NaN values.

    Parameters:
    time_series (np.ndarray): The input time series data.
    
    Returns:
    np.ndarray: The time series with outliers interpolated, preserving original NaNs.
    """
    if not isinstance(time_series, np.ndarray):
        raise ValueError("Input time series must be a numpy array")
    if time_series.ndim != 1:
        raise ValueError("Input time series must be one-dimensional")
    
    clean_series = time_series.astype(float)

    # Store the indices of the original NaN values
    original_nans = np.isnan(clean_series)

    if len(time_series) >= 3:
        # Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = np.nanpercentile(time_series, 25)
        Q3 = np.nanpercentile(time_series, 75)
        
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        
        # Determine the outlier boundaries
        lower_boundary = Q1 - iqr_factor * IQR
        upper_boundary = Q3 + iqr_factor * IQR
        
        # Identify outliers
        outliers = (time_series < lower_boundary) | (time_series > upper_boundary)
        
        # Set outliers to NaN for interpolation
        clean_series[outliers] = np.nan
        
        # Perform linear interpolation
        nans, x = np.isnan(clean_series), lambda z: z.nonzero()[0]
        clean_series[nans] = np.interp(x(nans), x(~nans), clean_series[~nans])
        
        # Restore original NaNs
        clean_series[original_nans] = np.nan
        
    return clean_series

def detect_and_interpolate_outliers_v2(time_series, window_size=3, std_factor=3):
    """
    Detects outliers in a time series using a rolling window-based adaptive method
    and interpolates them using linear interpolation, while preserving the original NaN values.

    Parameters:
    time_series (np.ndarray): The input time series data.
    window_size (int): The size of the rolling window for calculating local statistics.
    std_factor (float): The number of standard deviations to use for detecting outliers.

    Returns:
    np.ndarray: The time series with outliers interpolated, preserving original NaNs.
    """
    if not isinstance(time_series, np.ndarray):
        raise ValueError("Input time series must be a numpy array")
    if time_series.ndim != 1:
        raise ValueError("Input time series must be one-dimensional")

    clean_series = time_series.astype(float)

    # Store the indices of the original NaN values
    original_nans = np.isnan(clean_series)

    if len(time_series) >= window_size:
        # Calculate rolling median and rolling standard deviation
        rolling_median = pd.Series(clean_series).rolling(window=window_size, center=True).median()
        rolling_std = pd.Series(clean_series).rolling(window=window_size, center=True).std()

        # Fill initial and final missing rolling_median and rolling_std with the nearest valid value
        rolling_median = rolling_median.bfill().ffill()
        rolling_std = rolling_std.bfill().ffill()

        # Identify outliers based on deviation from the rolling median
        deviation = np.abs(clean_series - rolling_median)
        outliers = deviation > std_factor * rolling_std

        # Set outliers to NaN for interpolation
        try:
            clean_series[outliers] = np.nan

            # Perform linear interpolation
            nans, x = np.isnan(clean_series), lambda z: z.nonzero()[0]
            if np.any(~nans):
                f = interp1d(x(~nans & ~original_nans), clean_series[~nans & ~original_nans], kind='cubic', fill_value="extrapolate")
                clean_series[nans & ~original_nans] = f(x(nans & ~original_nans))

                # clean_series[nans] = np.interp(x(nans), x(nans), clean_series[nans])

            # Restore original NaNs
            clean_series = pd.Series(clean_series).rolling(window=3, center=True).mean().bfill().ffill()
            clean_series[original_nans] = np.nan
        except:
            pass
    return np.array(clean_series)

def normalize_time_and_resample(data, times, start_time=None, end_time=None, number_target_time_points=1000, ):
    """
    Normalize the time and resample the array based on the target time points.

    Parameters:
        data (ndarray): A 2D array where each row represents data at a specific frequency, 
                        and columns represent time points.
        times (ndarray): 1D array of original time points corresponding to the columns of `data`.
        start_time (float): Start time for normalization.
        end_time (float): End time for normalization.
        number_target_time_points (int): numbe of target time points for resampling.

    Returns:
        resampled_data (ndarray): The resampled array at target time points, shape 
                                  (data.shape[0], len(target_time_points)).
    """
    if data.ndim != 2:
        data = data.reshape(1, -1) # Reshape to 2D array if it's 1D like a single frequency band

    if start_time is None:
        start_time = times[0]
    if end_time is None:
        end_time = times[-1]

    # Normalize time to range [0, 1]
    normalized_time = (times - start_time) / (end_time - start_time)
    target_time_points = np.linspace(0, 1, number_target_time_points)
    
    # Replace NaN values in data with 0 (NOTE: Might need to change this)
    data = np.nan_to_num(data, nan=0.0)
    
    # Resample data to target time points using linear interpolation

    # NOTE: the resampled_data will be initiated as all zeros - Might need to change this
    resampled_data = np.zeros((data.shape[0], len(target_time_points)))
    
    for i in range(data.shape[0]):  # Iterate over each frequency
        interp_func = interp1d(
            normalized_time, data[i, :], kind='linear', fill_value="extrapolate"
        )
        resampled_data[i, :] = interp_func(target_time_points)

    return resampled_data

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = array + 1e-16
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

import numpy as np

def get_ordered_states(transition_matrix):
    """
    Orders states based on the most likely sequence of transitions.
    
    Args:
        transition_matrix (numpy.ndarray): Square matrix with transition probabilities.

    Returns:
        ordered_indices (list): List of state indices in the preferred order.
    """
    num_states = transition_matrix.shape[0]
    visited = set()
    ordered_indices = []

    # Start from the state with the highest outgoing probability sum
    start_state = np.argmax(transition_matrix.sum(axis=0))  
    current_state = start_state

    while len(ordered_indices) < num_states:
        ordered_indices.append(current_state)
        visited.add(current_state)

        # Find the most probable next state (excluding already visited states)
        next_states = np.argsort(transition_matrix[current_state])[::-1]  # Sort by probability (descending)
        next_state = next((s for s in next_states if s not in visited), None)

        if next_state is not None:
            current_state = next_state
        else:
            # If no unvisited state remains with direct high probability, pick any remaining state
            remaining_states = [s for s in range(num_states) if s not in visited]
            if remaining_states:
                current_state = remaining_states[0]  # Pick first remaining state

    return ordered_indices

def wquantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Compute the **weighted q‑quantile** of an array.

    The function ignores any element where either the value or its weight is
    NaN, then sorts the remaining data by value and accumulates weights until
    the desired quantile is reached.

    Parameters
    ----------
    values : np.ndarray
        Array of data points.
    weights : np.ndarray
        Corresponding non‑negative weights for each data point. Must be the same
        shape as `values`.
    q : float
        Desired quantile in the closed interval ``[0, 1]`` (e.g., ``0.5`` for
        the median).

    Returns
    -------
    float
        The weighted quantile. If `values` is empty after NaN removal or the
        total weight is zero, returns ``np.nan``.

    Notes
    -----
    The implementation is equivalent to the algorithm described in
    *Hyndman & Fan (1996), “Sample Quantiles in Statistical Packages,”*
    Method 2 for weighted data:

    1. Drop any pair where either entry is NaN.
    2. Sort the remaining values; carry the weights along.
    3. Let ``W`` be the cumulative sum of sorted weights and
       ``W_total`` the sum of all weights.
    4. The q‑quantile is the first value whose cumulative weight
       satisfies ``W ≥ q × W_total``.

    Examples
    --------
    >>> import numpy as np
    >>> wquantile(np.array([1, 2, 3]), np.array([1, 1, 1]), 0.5)
    2.0
    >>> wquantile(np.array([1, 2, 3]), np.array([0.1, 0.8, 0.1]), 0.5)
    2.0
    """

    v, w = map(np.asarray, (values, weights))
    msk  = ~(np.isnan(v) | np.isnan(w))
    v, w = v[msk], w[msk]

    if v.size == 0 or w.sum() == 0:
        return np.nan
    
    order = np.argsort(v)
    v_sorted, w_sorted = v[order], w[order]
    cum_w = np.cumsum(w_sorted)
    return v_sorted[cum_w >= q * w_sorted.sum()][0]

def weighted_pca(
    X: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_components: Optional[int] = None,
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inverse-probability-weighted PCA.

    Parameters
    ----------
    X : array, shape (n_features, n_samples)
        Data matrix with samples (time windows) in columns.
    weights : array, shape (n_samples,), optional
        IPW weights for each column.  If None, ordinary PCA is performed.
    n_components : int or None
        How many PCs to return. None -> keep all.
    center : bool
        Subtract the weighted mean before PCA (recommended).

    Returns
    -------
    scores : array, shape (n_components, n_samples)
        Principal-component time series (each row is a PC).
    components_ : array, shape (n_components, n_features)
        Loading vectors (same orientation as scikit-learn: rows = PCs).
    eigvals_ : array, shape (n_components,)
        Eigenvalues of the weighted covariance matrix.
    explained_variance_ratio_ : array, shape (n_components,)
        Fraction of variance explained by each PC.
    """
    F, N = X.shape
    if weights is None:
        weights = np.ones(N)
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != N:
        raise ValueError("weights must have length equal to X.shape[1]")
   
    # --- normalise for numerical stability (mean 1) ----------
    weights = weights / weights.mean()

    # --- (1) weighted mean & centring ------------------------
    if center:
        mu = (X * weights).sum(axis=1, keepdims=True) / weights.sum()
        Xc = X - mu
    else:
        Xc = X.copy()


    # --- (2) weighted covariance via square-root trick -------
    Wsqrt = np.sqrt(weights)          # shape (N,)
    Xw = Xc * Wsqrt                   # broadcasts across rows


    # --- (3) economy SVD (eig-decomp of covariance) ---------
    #     Xw = U S Vᵀ, cov = (1/Σw) * U S² Uᵀ
    U, S, _ = np.linalg.svd(Xw, full_matrices=False)
    eigvals = (S**2) / weights.sum()       # shape (min(F,N),)


    # --- (4) order & truncate -------------------------------
    if n_components is None or n_components > U.shape[1]:
        n_components = U.shape[1]
    components_ = U[:, :n_components].T        # PCs x features
    scores = (components_ @ Xc).T              # samples × PCs
    eigvals_ = eigvals[:n_components]
    explained_variance_ratio_ = eigvals_ / eigvals.sum()

    # ------ enforce positive orientation --------------------
    for k in range(n_components):
        comp = components_[k]
        pivot = np.argmax(np.abs(comp)) # index of largest amplitude 
        if comp[pivot] < 0:
            components_[k] *= -1
            scores[:, k] *= -1

    return scores, components_, eigvals_, explained_variance_ratio_


def gen_significance_string(p_value, marker='*'):
    if p_value < 0.001:
        return marker * 3
    elif p_value < 0.01:
        return marker * 2
    elif p_value < 0.05:
        return marker
    else:
        return ' '

def _weighted_box_stats(x, w=None, whis=1.5):
    """Return dict usable by matplotlib.Axes.bxp().

    Parameters
    ----------
    x : array-like (1D)
    w : array-like or None (same length as x)
    whis : float, Tukey multiplier (1.5 => default IQR rule)

    Notes
    -----
    * Weighted quartiles via supplied wquantile() when w provided.
    * Whiskers computed by Tukey fences from (weighted) Q1/Q3 but applied to raw x.
    """
    x = np.asarray(x, float)
    mask = np.isfinite(x)
    x = x[mask]
    if w is not None:
        w = np.asarray(w, float)[mask]
        s = w.sum()
        if s > 0:
            w = w / s
        med = wquantile(x, w, 0.5)
        q1 = wquantile(x, w, 0.25)
        q3 = wquantile(x, w, 0.75)
    else:
        q1, med, q3 = np.percentile(x, [25, 50, 75])

    iqr = q3 - q1
    lo_fence = q1 - whis * iqr
    hi_fence = q3 + whis * iqr

    in_fence = x[(x >= lo_fence) & (x <= hi_fence)]
    if in_fence.size:
        whisk_lo = np.min(in_fence)
        whisk_hi = np.max(in_fence)
    else:  # degenerate
        whisk_lo = q1
        whisk_hi = q3

    fliers = x[(x < whisk_lo) | (x > whisk_hi)]
    return {
        'med': med,
        'q1': q1,
        'q3': q3,
        'whislo': whisk_lo,
        'whishi': whisk_hi,
        'fliers': fliers,
    }

def _paired_perm_p(diffs, weights, n_perm=10000, rng=None):
    """
    Weighted paired test via sign-flip permutation on 'diffs' (post - pre).
    Test statistic: weighted median of diffs.
    Two-sided p.
    """
    if rng is None:
        rng = np.random.default_rng()
    diffs = np.asarray(diffs, float)
    weights = np.asarray(weights, float)
    # normalize weights (safe if all zero? -> fall back to equal)
    s = weights.sum()
    if s <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / s
    obs = wquantile(diffs, weights, 0.5)

    flips = rng.choice([-1.0, 1.0], size=(n_perm, diffs.size))
    perm_stats = np.apply_along_axis(
        lambda v: wquantile(diffs * v, weights, 0.5),
        axis=1,
        arr=flips
    )
    p = (np.sum(np.abs(perm_stats) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


def _two_sample_perm_p(x, w_x, y, w_y, n_perm=10000, rng=None):
    """
    Weighted two-sample test for difference in medians.
    Pools data, permutes group labels (respecting total n), re-computes
    weighted median difference each perm. Two-sided p.

    Returns (obs_stat, p).
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, float); y = np.asarray(y, float)
    w_x = np.asarray(w_x, float); w_y = np.asarray(w_y, float)

    # drop NaNs
    mx = np.isfinite(x); my = np.isfinite(y)
    x = x[mx]; w_x = w_x[mx]
    y = y[my]; w_y = w_y[my]

    if x.size == 0 or y.size == 0:
        return np.nan, np.nan

    # normalize per group
    sx = w_x.sum(); sy = w_y.sum()
    if sx <= 0: w_x = np.ones_like(w_x) / len(w_x)
    else:       w_x = w_x / sx
    if sy <= 0: w_y = np.ones_like(w_y) / len(w_y)
    else:       w_y = w_y / sy

    med_x = wquantile(x, w_x, 0.5)
    med_y = wquantile(y, w_y, 0.5)
    obs = med_x - med_y

    # pool
    pooled_vals = np.concatenate([x, y])
    pooled_w    = np.concatenate([w_x, w_y])
    n_x = x.size
    n_tot = pooled_vals.size

    # precompute normalized weights for perm splits efficiently
    idx = np.arange(n_tot)
    perm_stats = np.empty(n_perm, float)
    for i in range(n_perm):
        rng.shuffle(idx)            # in-place
        sel = idx[:n_x]
        mask = np.zeros(n_tot, bool)
        mask[sel] = True

        xv = pooled_vals[mask]; xw = pooled_w[mask]
        yv = pooled_vals[~mask]; yw = pooled_w[~mask]

        sx = xw.sum(); sy = yw.sum()
        if sx <= 0: xw = np.ones_like(xw) / len(xw)
        else:       xw = xw / sx
        if sy <= 0: yw = np.ones_like(yw) / len(yw)
        else:       yw = yw / sy

        perm_stats[i] = wquantile(xv, xw, 0.5) - wquantile(yv, yw, 0.5)

    p = (np.sum(np.abs(perm_stats) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


# ======================================================================================
# GEE per band; fitting, omnibus tests, and cellwise contrasts 
# ======================================================================================

#------------- fit GEE on long dataframe---------------------------------------------
def _fit_gee_on_df(
        df_long,
        *,
        formula: str | None = None
    ):
    """
    Expects columns: y, subj, group (A/B), segment, region, w
    segment and region should be categorical (or will be treated as such via C()).
    """

    # --- formula
    if formula is None:
        formula = "y ~ C(group) * C(segment) * C(region)"

    gee = sm.GEE.from_formula(
        formula=formula,
        groups="subj",
        data=df_long,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
        weights=df_long["w"]
    )
    return gee.fit()

#--------------Fit a single GEE for one frequency band across all regions -----------
def fit_gee_group_time_region(
    formula,
    region_names,
    A_by_region, 
    B_by_region,          # dict: region -> (n_subj x n_seg) arrays
    ipw_A=None, 
    ipw_B=None,            # 1D arrays length n_subj
    subtract_pre=False,
    pre_A_by_region=None, 
    pre_B_by_region=None,
    region_order=None,
    clip_weights=True
):
    """
    Fit one GEE per band with outcome y = (post bandpower [dB]) or (post - pre) if subtract_pre=True.
    Model: y ~ C(group) * C(segment) * C(region), clustered by subject ('subj').
    Weights are stabilized IPWs if provided; otherwise unit weights.

    Returns
    -------
    res : GEEResults
        Fitted GEE result.
    df_long : pd.DataFrame
        Long-format dataframe used for fitting (y, group, segment, region, subj, w).
    """
    dfs = []

    # sanity checks
    nA, nSeg = next(iter(A_by_region.values())).shape
    nB, nSegB = next(iter(B_by_region.values())).shape
    assert nSeg == nSegB, "A/B must have same #segments"

    # stabilized weights (subject level)
    if (ipw_A is not None) or (ipw_B is not None):
        wA = np.asarray(ipw_A) if ipw_A is not None else np.ones(nA)
        wB = np.asarray(ipw_B) if ipw_B is not None else np.ones(nB)
        wA = wA / np.nanmean(wA); wB = wB / np.nanmean(wB)
        if clip_weights:
            loA, hiA = np.nanpercentile(wA, [1, 99]); wA = np.clip(wA, loA, hiA)
            loB, hiB = np.nanpercentile(wB, [1, 99]); wB = np.clip(wB, loB, hiB)
    else:
        wA = np.ones(nA); wB = np.ones(nB)

    # build long df for each region, continuing subject IDs across groups
    for region in region_names:
        A_mat = np.asarray(A_by_region[region])
        B_mat = np.asarray(B_by_region[region])

        if subtract_pre:
            if pre_A_by_region is None or pre_B_by_region is None:
                raise ValueError("Provide pre_A_by_region/pre_B_by_region when subtract_pre=True.")
            A_mat = A_mat - np.asarray(pre_A_by_region[region])[:, None]
            B_mat = B_mat - np.asarray(pre_B_by_region[region])[:, None]
        
        # Group A
        dfA = pd.DataFrame({
            "y": A_mat.ravel(),
            "subj": np.repeat(np.arange(nA), nSeg),
            "group": "A",
            "segment": np.tile(np.arange(nSeg), nA),
            "region": region,
            "w": np.repeat(wA, nSeg),
        })
        
        # Group B (subject ids continue)
        dfB = pd.DataFrame({
            "y": B_mat.ravel(),
            "subj": np.repeat(np.arange(nB)+nA, nSeg),
            "group": "B",
            "segment": np.tile(np.arange(nSeg), nB),
            "region": region,
            "w": np.repeat(wB, nSeg),
        })
        dfs.append(dfA); dfs.append(dfB)

    df_long = pd.concat(dfs, ignore_index=True)


    # Explicit, stable reference levels
    group_type  = CategoricalDtype(categories=['A','B'], ordered=True)  # ref='A'
    
    # Make sure segment and region orders are what you intend
    seg_cats    = sorted(pd.unique(df_long['segment']))
    seg_type    = CategoricalDtype(categories=seg_cats, ordered=True)

    # region_cats = list(region_names)  # keep your specified order as reference

    if region_order is None:
        region_cats = list(region_names)
    else:
        # put requested order first, then append any remaining regions
        rest = [r for r in region_names if r not in region_order]
        region_cats = list(region_order) + rest
    region_type = CategoricalDtype(categories=region_cats, ordered=True)

    df_long['group']  = df_long['group'].astype(group_type)
    df_long['segment']= df_long['segment'].astype(seg_type)
    df_long['region'] = df_long['region'].astype(region_type)
    
    res = _fit_gee_on_df(df_long, formula=formula)

    return res, df_long

#--------------------------------------------------------------------------------
# Permutation cache: fit permuted GEE models ONCE, reuse everywhere
#--------------------------------------------------------------------------------

# Do one subject-level group label permutation, preserving group sizes ----------
def _permute_subject_groups(df_long, rng, n_A):
    """
    Returns a new DataFrame with group labels permuted across subjects,
    preserving n_A and n_B. IPWs in column 'w' are left unchanged.
    """
    df_perm = df_long.copy(deep=True)

    subj_group = df_long.groupby('subj')['group'].first()
    subj_ids = subj_group.index.to_numpy()

    # draw a permutation of subject IDs and assign the first n_A to 'A'
    perm = rng.permutation(subj_ids)
    new_A = set(perm[:n_A])

    # keep original categorical dtypes (and references)
    group_type  = df_long['group'].dtype
    seg_type    = df_long['segment'].dtype
    region_type = df_long['region'].dtype

    df_perm['group']  = pd.Categorical(np.where(df_perm['subj'].isin(new_A), 'A', 'B'),
                                       categories=group_type.categories, ordered=True)
    df_perm['segment']= df_perm['segment'].astype(seg_type)
    df_perm['region'] = df_perm['region'].astype(region_type)

    return df_perm

# --------- precompute permuted GEE fits for one band ---------------------------
def prefit_permuted_gees_for_band(formula, df_long, n_perm=1000, seed=None, verbose=True):
    """
    Fit the permuted GEE models ONCE and return a list of GEEResults.
    Any permutation that fails to converge is skipped.

    Returns: list_of_results (length = n_perm_used), rng_state dict
    """
    rng = np.random.default_rng(seed)

    subj_group = df_long.groupby('subj')['group'].first()
    n_A = int((subj_group == 'A').sum())

    perm_fits = []
    n_tried = 0
    while len(perm_fits) < n_perm:
        n_tried += 1
        try:
            dfp = _permute_subject_groups(df_long, rng, n_A)
            resp = _fit_gee_on_df(dfp, formula=formula)
            # quick sanity check on parameter names shape (optional)
            if resp is not None and resp.params is not None:
                perm_fits.append(resp)
        except Exception:
            # convergence/numerical hiccup; skip and continue
            continue

        if verbose and (len(perm_fits) % max(1, n_perm // 10) == 0):
            print(f"  ...perm fits cached: {len(perm_fits)} / {n_perm} (tries={n_tried})")

    return perm_fits, {"seed": seed, "n_tried": n_tried, "n_used": len(perm_fits)}


# ------------------------------------------------------------------------------
# Omnibus Wald tests
# ------------------------------------------------------------------------------
# Build L and do joint Wald from a selector (names -> keep?)------------------------
def _build_L_from_selector(res, keep):
    names = list(res.params.index)
    p_dim = len(names)
    idx = [i for i, nm in enumerate(names) if keep(nm)]
    if not idx:
        return None, [], names
    L = np.zeros((len(idx), p_dim))
    for r, i in enumerate(idx):
        L[r, i] = 1.0
    return L, [names[i] for i in idx], names

def _wald_stat_on(res, L):
    """
    Wald chi-square for contrast matrix L on fitted result `res` (scalar test).
    Returns (chi2, df, p).
    """
    wtest = res.wald_test(L, scalar=True)
    chi2 = float(np.asarray(getattr(wtest, "statistic", np.nan)).reshape(()))
    pval = float(np.asarray(getattr(wtest, "pvalue",    np.nan)).reshape(()))
    df_attr = None
    for attr in ("df_denom", "df_num", "df"):
        if hasattr(wtest, attr) and (getattr(wtest, attr) is not None):
            try:
                df_attr = int(np.asarray(getattr(wtest, attr)).reshape(()))
            except Exception:
                df_attr = None
            break
    return chi2, df_attr, pval

def _wald_from_L_with_cached_perm(res_obs, L, perm_fits=None):
    """
    Wald for arbitrary contrast matrix L (possibly multi-row).
    Reuses cached permuted fits (if provided) for permutation p-values.
    Returns dict: {'p_wald','chi2','df','k_constraints','p_perm','n_perm_used'}
    """
    # L, kept_names, _ = _build_L_from_selector(res_obs, keep)
    if L is None:
        return {'p_wald': np.nan, 'chi2': np.nan, 'df': None,
                'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0}

    chi2, df, p_wald = _wald_stat_on(res_obs, L)
    out = {
        'p_wald': p_wald,
        'chi2': chi2,
        'df': df,
        'k_constraints': int(L.shape[0]),
        # 'names': kept_names,
        'p_perm': np.nan,
        'n_perm_used': 0
    }

    # Permutation p using cached permuted fits
    if (perm_fits is not None) and np.isfinite(chi2):
        obs = float(chi2)
        ge = 0
        used = 0
        for rp in perm_fits:
            obs_names = res_obs.params.index
            rp_names  = rp.params.index
            if not rp_names.equals(obs_names):
                # Reorder L's columns from OBS order into RP order
                col_idx = [obs_names.get_loc(nm) for nm in rp_names]
                Lp = L[:, col_idx]
            else:
                Lp = L
            stat, _, _ = _wald_stat_on(rp, Lp) #Same L on permuted fit
            if not np.isfinite(stat):
                continue
            used += 1
            if stat >= obs - 1e-12:
                ge += 1
        out['n_perm_used'] = used
        out['p_perm'] = (ge + 1) / (used + 1) if used > 0 else np.nan
    return out

def _wald_from_selector(res_obs, keep, perm_fits=None):
    L, kept_names, _ = _build_L_from_selector(res_obs, keep)
    out = _wald_from_L_with_cached_perm(res_obs, L, perm_fits=perm_fits)
    out['names'] = kept_names
    return out

# -------- helpers: categories & cell selector ---------------------
def _get_segment_region_categories(df_long):
    """Return (seg_cats, region_cats, seg_ref, region_ref) in coded order."""
    seg_cats    = list(df_long['segment'].cat.categories) if hasattr(df_long['segment'], 'cat') else sorted(df_long['segment'].unique())
    region_cats = list(df_long['region'].cat.categories)  if hasattr(df_long['region'],  'cat')  else sorted(df_long['region'].unique())
    seg_ref, region_ref = seg_cats[0], region_cats[0]
    return seg_cats, region_cats, str(seg_ref), str(region_ref)

def _cell_keep_selector(res, seg_label, region_label, seg_ref, region_ref):
    """
    Returns a predicate keep(name) that selects exactly the coefficients
    that sum to the group (B−A) effect at (seg_label, region_label),
    under references: group='A', segment=seg_ref, region=region_ref.

    Adaptive selector that works for both full and parsimonious models.
    Checks which parameters actually exist in the fitted model and includes only those.
    
    Potential parameters:
    - C(group)[T.B] (always present)
    - C(group)[T.B]:C(segment)[T.seg_label] (if seg != ref and group×segment interaction exists)
    - C(group)[T.B]:C(region)[T.region_label] (if region != ref and group×region interaction exists)
    - C(group)[T.B]:C(segment)[T.seg_label]:C(region)[T.region_label] (if both != ref and 3-way exists)

    """
    # needed = {'group[T.B]'}
    # if seg_label != seg_ref:
    #     needed.add(f'group[T.B]:C(segment)[T.{seg_label}]')
    # if region_label != region_ref:
    #     needed.add(f'group[T.B]:C(region)[T.{region_label}]')
    # if (seg_label != seg_ref) and (region_label != region_ref):
    #     needed.add(f'group[T.B]:C(segment)[T.{seg_label}]:C(region)[T.{region_label}]')
    # return lambda nm: nm in needed
    param_names = set(res.params.index)
    needed = set()
    
    # Main group effect (should always exist)
    if 'C(group)[T.B]' in param_names:
        needed.add('C(group)[T.B]')
    
    # Group×segment interaction (if exists and seg != ref)
    if seg_label != seg_ref:
        group_seg_param = f'C(group)[T.B]:C(segment)[T.{seg_label}]'
        if group_seg_param in param_names:
            needed.add(group_seg_param)
    
    # Group×region interaction (if exists and region != ref)
    if region_label != region_ref:
        group_region_param = f'C(group)[T.B]:C(region)[T.{region_label}]'
        if group_region_param in param_names:
            needed.add(group_region_param)
    
    # 3-way interaction (if exists and both != ref)
    if (seg_label != seg_ref) and (region_label != region_ref):
        three_way_param = f'C(group)[T.B]:C(segment)[T.{seg_label}]:C(region)[T.{region_label}]'
        if three_way_param in param_names:
            needed.add(three_way_param)
    
    return lambda nm: nm in needed


# -------- build region-focused omnibus L --------------------------
def build_L_region_any_segment(res, df_long, region_label):
    """
    For a fixed region, stack the K per-segment group-contrast rows into a K×p matrix L.
    Each row corresponds to (B−A) at that segment within the given region.
    """
    seg_cats, region_cats, seg_ref, region_ref = _get_segment_region_categories(df_long)
    # sanity on region
    if str(region_label) not in map(str, region_cats):
        raise ValueError(f"region_label '{region_label}' not found in categories {region_cats}")

    L_rows = []
    for seg in seg_cats:
        keep = _cell_keep_selector(res, seg_label=str(seg), region_label=str(region_label),
                                   seg_ref=seg_ref, region_ref=region_ref)
        L_cell, kept_names, _ = _build_L_from_selector(res, keep)
        if L_cell is None:
            # no terms matched; skip this segment (rare)
            continue
        # Collapse the kept rows to a single 1×p row for this cell
        L_rows.append(L_cell.sum(axis=0, keepdims=True))

    if not L_rows:
        return None  # nothing to test for this region

    # Stack to K×p
    L = np.vstack(L_rows)
    return L

def build_L_segment_any_region(res, df_long, seg_label, region_subset=None):
    """
    For a fixed time segment, stack the K per-region group-contrast rows into a Kxp matrix L.
    Each row corresponds to (B-A) at that region within the given segment 
    """
    seg_cats, region_cats, seg_ref, region_ref = _get_segment_region_categories(df_long)

    if (region_subset is None) or (region_subset == "all"):
        region_subset = region_cats

    # sanity on segment 
    if str(seg_label) not in map(str, seg_cats):
        raise ValueError(f"segment_label '{seg_label}' not found in categories {seg_cats}")
    
    L_rows = []
    for region in region_subset:
        keep = _cell_keep_selector(res, seg_label=str(seg_label), region_label=str(region),
                                   seg_ref=seg_ref, region_ref=region_ref)
        L_cell, keep_names, _ = _build_L_from_selector(res, keep)

        if L_cell is None:
            # no terms matched; skip this region (rare)
            continue 
        # Collapse the lept rows to a single 1xp row for this cell
        L_rows.append(L_cell.sum(axis=0, keepdims=True))

    if not L_rows:
        return None # nothing to test for this segment 
    
    # Stack to Kxp
    L = np.vstack(L_rows)
    return L

def _build_L_all_cells(res, df_long, region_subset=None):
    """
    Stack one 1×p row per (segment, region) cell for the B−A contrast.
    Returns an (K*R_sel)×p matrix. Requires df_long.
    """
    if df_long is None:
        raise ValueError("_build_L_all_cells requires df_long")

    seg_cats, region_cats, seg_ref, region_ref = _get_segment_region_categories(df_long)

    if (region_subset is None) or (region_subset == "all"):
        region_subset = list(map(str, region_cats))
    else:
        region_subset = [str(r) for r in region_subset]
        missing = [r for r in region_subset if r not in map(str, region_cats)]
        if missing:
            raise ValueError(f"Regions not found in data: {missing}. Available: {list(map(str, region_cats))}")

    rows = []
    for seg in seg_cats:
        for reg in region_subset:
            keep = _cell_keep_selector(res, seg_label=str(seg), region_label=str(reg),
                                       seg_ref=seg_ref, region_ref=region_ref)
            L_cell, _, _ = _build_L_from_selector(res, keep)
            if L_cell is None or L_cell.size == 0:
                continue
            rows.append(L_cell.sum(axis=0, keepdims=True))  # 1×p row for this cell

    if not rows:
        return None
    return np.vstack(rows)  # (K * len(region_subset)) × p

def omnibus_any_group(res, df_long=None, region_subset=None, perm_fits=None, *, kind="mean"):
    """
    Joint Wald test for ANY group effect anywhere.
    Q: Is there evidence that Group B differs from Group A anywhere (any segment, any region)?
    
    This tests MAIN + INTERACTION effects - whether there's any group difference anywhere,
    including the main group effect at reference cell and all group interactions.
    
    This is fundamentally different from interaction-only tests:
    - omnibus_any_group: Tests if group effects EXIST anywhere
    - omnibus_group_x_*: Tests if group effects VARY across conditions (interaction only)

    - Default (region_subset=None or 'all'): tests all coefficients involving 'group'
      (main effect + all group×segment, group×region, group×segment×region interactions).
    - If region_subset is provided (e.g., ['frontal','parietal']), tests whether there's 
      any group effect in any segment within those specific regions only.
      Includes both main effects and interactions within the specified regions.

    Update:
    kind='any'  (default): tests ANY coefficient involving 'group' (main + interactions).
    kind='mean': tests the AVERAGED group effect (B−A) across segments and regions (df=1).

    """
    kind = kind.lower()
    if kind not in {'any', 'mean'}:
        raise ValueError("kind must be 'any' or 'mean'")

    if df_long is None:
            raise ValueError("omnibus_any_group(..., regions=...) requires df_long.")

    if kind == "any":
        if (region_subset is None) or (region_subset == 'all'):
            # Test all coefficients involving 'group'(main effect + all interactions)
            return _wald_from_selector(res, lambda nm: 'C(group)[' in nm, perm_fits=perm_fits)

        # region-restricted: build L by stacking per-segment rows for the chosen regions
        
        region_subset = [str(r) for r in region_subset]
        seg_cats, region_cats, seg_ref, region_ref = _get_segment_region_categories(df_long)
        region_cats_str = list(map(str, region_cats))
        
        # Validate that all requested regions exist
        missing_regions = [r for r in region_subset if r not in region_cats_str]
        if missing_regions:
            raise ValueError(f"Regions not found in data: {missing_regions}. Available: {region_cats_str}")

        L_blocks = []
        for reg in region_subset:
            L_reg = build_L_region_any_segment(res, df_long, region_label=reg) #Kxp
            if L_reg is not None and L_reg.size:
                L_blocks.append(L_reg)

        if not L_blocks:
            # No valid regions found; return empty result
            return {'p_wald': np.nan, 'chi2': np.nan, 'df': None, 'k_constraints': 0,
                    'names': [], 'p_perm': np.nan, 'n_perm_used': 0}

        L = np.vstack(L_blocks)  # stack all selected regions' segments
        out = _wald_from_L_with_cached_perm(res, L, perm_fits=perm_fits)
        out['names'] = []  # could populate with param names if desired

        return out
    
    else:
        # --- averaged main effect across segments and (optionally subset of) regions ---
        L_all = _build_L_all_cells(res, df_long, region_subset=region_subset)  # (m)×p
        if L_all is None or L_all.size == 0:
            return {'p_wald': np.nan, 'chi2': np.nan, 'df': None, 'k_constraints': 0,
                    'names': [], 'p_perm': np.nan, 'n_perm_used': 0}

        # # Row-mean → 1×p contrast (df=1)
        # # Count how many times each coefficient contributes across cells
        # # This correctly handles main effect, 2-way, 3-way interactions
        # # (reference levels contribute zero)
        # weights = (L_all != 0).astype(float)
        # weights_sum = weights.sum(axis=0)  # how many cells each coeff contributes to
        # L_mean = L_all.sum(axis=0, keepdims=True)  # sum across all cells
        # L_mean = L_mean / weights_sum  # divide each coeff by number of contributing cells
        
        L_mean = L_all.mean(axis=0, keepdims=True)

        # Convert NaNs (from reference coefficients not appearing) to zero
        L_mean = np.nan_to_num(L_mean)

        out = _wald_from_L_with_cached_perm(res, L_mean, perm_fits=perm_fits)
        out['names'] = []        # could list contributing params if you like
        out['k_constraints'] = 1 # single averaged contrast

        return out

def _has_interaction_terms(res, interaction_type):
    """
    Check if model has specific interaction terms.
    
    interaction_type options:
    - 'group_segment': group×segment interactions
    - 'group_region': group×region interactions  
    - 'group_segment_region': 3-way interactions
    """
    param_names = set(res.params.index)
    
    if interaction_type == 'group_segment':
        return any('C(group)[T.B]:C(segment)[T.' in nm for nm in param_names)
    elif interaction_type == 'group_region':
        return any('C(group)[T.B]:C(region)[T.' in nm for nm in param_names)
    elif interaction_type == 'group_segment_region':
        return any('C(group)[T.B]:C(segment)[T.' in nm and 'C(region)[T.' in nm for nm in param_names)
    else:
        raise ValueError(f"Unknown interaction type: {interaction_type}")

def omnibus_group_x_segment(res, df_long=None, region_subset=None, perm_fits=None):
    """
    Joint Wald test of group×segment interaction effects.
    Q: Does the caffeine-placebo difference vary by segment (time), on average across regions?

    H0 (null): The between-group gap (B - A) is identical across all segments, averaged over the selected regions
    
    This tests INTERACTION effects only - how the group effect varies by segment,
    excluding the main group effect at the reference segment/region.

    
    - Default (region_subset=None or 'all'): tests all coefficients involving group×segment
      interactions, including 3-way group×segment×region terms.
    - If region_subset is provided, tests whether the group effect differs between 
      each segment and the reference segment, within the specified region(s).
      Reference segment is correctly excluded as we're testing interaction (temporal
      modulation of group effect), not main effects.

    - Returns empty results if no group x segment interaction exist.
    """
    # return _wald_from_selector(res, lambda nm: ('C(group)[' in nm) and ('C(segment)' in nm), perm_fits=perm_fits)
    if not _has_interaction_terms(res, 'group_segment'):
        return {
            'p_wald': np.nan, 'chi2': np.nan, 'df': None,
            'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0
        }

    if df_long is None:
        raise ValueError("df_long is required when region_subset is provided.")


    seg_cats, region_cats, seg_ref, region_ref = _get_segment_region_categories(df_long)
    region_cats_str = list(map(str, region_cats))
    if (region_subset is None) or (region_subset == 'all'):
        region_subset = region_cats_str
    else:
        region_subset = [str(r) for r in region_subset]
        missing_regions = [r for r in region_subset if r not in region_cats_str]
        if missing_regions:
            raise ValueError(f"Regions not found: {missing_regions}. Available: {region_cats_str}")

    # Build L for each segment (relative to reference segment) averaged over regions
    rows = []
    for seg in seg_cats:
        if str(seg) == str(seg_ref):
            # skip reference segment - we're testing INTERACTION effects
            # (how segments differ from reference), not main effects at reference
            continue
        
        # Collect segment difference for each region
        seg_diffs = []
        for region_label in region_subset:
            if region_label not in region_cats_str:
                continue

            keep_seg = _cell_keep_selector(
                res,
                seg_label=str(seg), 
                region_label=str(region_label),
                seg_ref=seg_ref, 
                region_ref=region_ref
            )
            L_seg, _, _ = _build_L_from_selector(res, keep_seg)
            if L_seg is None or L_seg.size == 0:
                continue 
            L_seg = L_seg.sum(axis=0, keepdims=True)

            # group effect at reference segment (for the same region)
            keep_ref = _cell_keep_selector(
                res, 
                seg_label=str(seg_ref),
                region_label=str(region_label),
                seg_ref=seg_ref, 
                region_ref=region_ref
            )

            L_ref, _, _ = _build_L_from_selector(res, keep_ref)
            if L_ref is None or L_ref.size == 0:
                continue
            L_ref = L_ref.sum(axis=0, keepdims=True)            

            # Test interaction: (group effect at segment) - (group effect at reference) = 0
            seg_diffs.append(L_seg - L_ref)

        if seg_diffs:
            # Average across regions
            L_avg = np.mean(np.vstack(seg_diffs), axis=0, keepdims=True)
            rows.append(L_avg)

    if not rows:
        # return an empty-shaped result in your usual format
        return {
            'p_wald': np.nan, 
            'chi2': np.nan, 
            'df': None,
            'k_constraints': 0,
            'names': [], 
            'p_perm': np.nan, 
            'n_perm_used': 0
        }
    
    L = np.vstack(rows) 
    out = _wald_from_L_with_cached_perm(res, L, perm_fits=perm_fits)
    out['names'] = []

    return out
   
def omnibus_group_x_region(res, df_long=None, region_subset=None, perm_fits=None):
    """
    Joint Wald test of group × region interaction effects (segment-averaged)
    Q: Does the caffeine-placebo difference vary by region, on average across segments?

    H0 (null):
      For every segment, the between-group gap (B - A) is identical across all regions
      and after averaging across segments the target regions do not differ from the reference.

    Behavior:
      - For each non-reference target region in `region_subset`, compute for every segment:
          (group effect at segment, target_region) - (group effect at same segment, reference_region)
        then average these per-segment contrasts across segments to obtain one 1×p row
        for that target region.
      - Stack these per-region averaged rows and run a joint Wald test.
      - df = number of (non-reference) regions included (or fewer if some regions have no valid rows).

    - Default (region_subset=None or 'all'): reproduces the original region-wide test.
    - If region_subset is provided (e.g., ['frontal', 'parietal']), tests within those regions only,
      comparing each to the reference region across segments. Requires df_long to identify categories.
    
    - Returns empty results if no group × region interactions exist.
    """
    # return _wald_from_selector(res, lambda nm: ('C(group)[' in nm) and ('C(region)' in nm), perm_fits=perm_fits)

    if not _has_interaction_terms(res, 'group_region'):
        return {
            'p_wald': np.nan, 'chi2': np.nan, 'df': None,
            'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0
        }

    if df_long is None:
        raise ValueError("df_long is required when region_subset is provided.")

    # categories and references
    seg_cats, region_cats, seg_ref, region_ref = _get_segment_region_categories(df_long)
    region_cats_str = list(map(str, region_cats))

    # region_subset handling (default = all regions)
    if (region_subset is None) or (region_subset == 'all'):
        region_subset = region_cats_str
    else:
        region_subset = [str(r) for r in region_subset]
        missing_regions = [r for r in region_subset if r not in region_cats_str]
        if missing_regions:
            raise ValueError(f"Regions not found: {missing_regions}. Available: {region_cats_str}")

    rows_per_region = [] # will hold 1xp arrays (one per target region)

    # For each target region (exclude the reference region)
    for target_region in region_subset:
        if str(target_region) == str(region_ref):
            continue

        seg_contrasts = []  # collect (L_cell - L_ref) for each segment for this region

        for seg in seg_cats:
            seg_label = str(seg)

            # Get L_cell: group effect at (seg, target_region)
            keep_cell = _cell_keep_selector(
                res,
                seg_label=seg_label,
                region_label=str(target_region),
                seg_ref=seg_ref,
                region_ref=region_ref,
            )
            L_cell, _, _ = _build_L_from_selector(res, keep_cell)
            if L_cell is None or L_cell.size == 0:
                continue
            L_cell = L_cell.sum(axis=0, keepdims=True)

            # Get L_ref: group effect at (seg, reference_region)
            keep_ref = _cell_keep_selector(
                res,
                seg_label=seg_label,
                region_label=str(region_ref),
                seg_ref=seg_ref,
                region_ref=region_ref,
            )
            L_ref, _, _ = _build_L_from_selector(res, keep_ref)
            if L_ref is None or L_ref.size == 0:
                # if reference row missing for this segment, skip this segment
                continue
            L_ref = L_ref.sum(axis=0, keepdims=True)

            # Subtract to get interaction contrast for this segment & target region
            seg_contrasts.append(L_cell - L_ref)
    
        if not seg_contrasts:
            # no valid segments for this target_region -> skip region
            continue
    
        # Average across segments to produce a single 1 x p row for this region
        L_region_avg = np.mean(np.vstack(seg_contrasts), axis=0, keepdims=True)
        rows_per_region.append((target_region, L_region_avg))

    if not rows_per_region:
        return {
            'p_wald': np.nan,
            'chi2': np.nan,
            'df': None,
            'k_constraints': 0,
            'names': [],
            'p_perm': np.nan,
            'n_perm_used': 0,
        }

    # Stack averaged rows (one per target region)
    L_list = [r[1] for r in rows_per_region]
    L = np.vstack(L_list)  # R' x p (R' = number of non-ref regions with data)

    out = _wald_from_L_with_cached_perm(res, L, perm_fits=perm_fits)
    out['names'] = [r[0] for r in rows_per_region]  # label rows by region

    return out

def omnibus_group_x_segment_x_region(res, df_long=None, region_subset=None, perm_fits=None):
    """
    NOTE: This needs word to make it compatible with the implementation of two-way interactions 

    Joint Wald test of 3-way group×segment×region interaction coefficients.
    Q: Does the segment pattern of the group effect differ across regions?
    
    This tests pure 3-way interaction effects: whether the group×segment interaction
    pattern varies by region (beyond any group×region or group×segment main interactions).
    
    - Default (region_subset=None or 'all'): tests all 3-way interaction terms across all regions.
    - If region_subset is provided (e.g., ['frontal', 'parietal']), tests only the 3-way 
      interactions involving those specific regions (excluding reference region, which 
      would represent lower-order interactions already tested elsewhere).
    
    - Returns empty results if no 3-way interactions exist.
    """
    # return _wald_from_selector(res, lambda nm: ('C(group)[' in nm) and ('C(segment)' in nm) and ('C(region)' in nm), perm_fits=perm_fits)
    if not _has_interaction_terms(res, 'group_segment_region'):
        return {
            'p_wald': np.nan, 'chi2': np.nan, 'df': None,
            'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0
        }
    
    if (region_subset is None) or (region_subset == 'all'):
        # Original behavior: test all 3-way interaction terms
        return _wald_from_selector(
            res, 
            lambda nm: ('C(group)[' in nm) and ('C(segment)' in nm) and ('C(region)' in nm), 
            perm_fits=perm_fits
        )

    if df_long is None:
        raise ValueError("df_long is required when region_subset is provided.")
    
    region_subset = [str(r) for r in region_subset]
    seg_cats, region_cats, seg_ref, region_ref = _get_segment_region_categories(df_long)
    region_cats_str = list(map(str, region_cats))

    # Validate that all requested regions exist
    missing_regions = [r for r in region_subset if r not in region_cats_str]
    if missing_regions:
        raise ValueError(f"Regions not found in data: {missing_regions}. Available: {region_cats_str}")

    # Build selector that only includes 3-way terms for the specified regions
    # Correctly excludes reference region AND reference segment since 3-way terms 
    # only exist for non-reference levels of both factors
    def region_specific_selector(nm):
        # Must be a 3-way interaction term
        if not (('C(group)[' in nm) and ('C(segment)' in nm) and ('C(region)' in nm)):
            return False
        
        # 3-way terms only exist for non-reference levels of BOTH segment and region
        # So we need both C(segment)[T.non_ref] AND C(region)[T.non_ref] in the term name
        
        # Check if this term involves any of our target regions (excluding reference)
        region_match = False
        for region_label in region_subset:
            if (region_label in region_cats_str and 
                region_label != region_ref and  # 3-way terms don't exist for reference region
                f'C(region)[T.{region_label}]' in nm):
                region_match = True
                break
        
        # Also check that it's not the reference segment (3-way terms need non-reference segment too)
        segment_match = any(f'C(segment)[T.{seg}]' in nm for seg in seg_cats if str(seg) != str(seg_ref))
        
        return region_match and segment_match

    return _wald_from_selector(res, region_specific_selector, perm_fits=perm_fits)

#-------- Print the Wald and (optionally) permutation stats and p-values-----------------------------------
def print_omnibus_summary(res_by_band, fband, df_long_by_band=None, perm_fits_by_band=None, region_subset=None):
    """
    Print omnibus tests for a band using prefit permuted GEEs.
    
    Parameters:
    - res_by_band: dict like {'alpha': gee_result, ...}
    - df_long_by_band: dict like {'alpha': df_long, ...} (required if region_subset provided)
    - perm_fits_by_band: dict like {'alpha': [gee_result_perm1, ...], ...} or None
    - region_subset: list of region names to focus on, or None for global tests
    """
    res = res_by_band[fband]
    perms = None if (perm_fits_by_band is None) else perm_fits_by_band.get(fband, None)
    dfl = None if (df_long_by_band is None) else df_long_by_band.get(fband, None)
    
    # Validate inputs when region_subset is provided
    if region_subset is not None and dfl is None:
        raise ValueError("df_long_by_band is required when region_subset is provided.")

    # Determine scope label for output
    scope_label = "Global" if region_subset is None else f"Region-subset {region_subset}"
    
    tests = {
        "ANY group term"       : omnibus_any_group(res, df_long=dfl, region_subset=region_subset, perm_fits=perms, kind='mean'),
        "group×segment"        : omnibus_group_x_segment(res, df_long=dfl, region_subset=region_subset, perm_fits=perms),
        "group×region"         : omnibus_group_x_region(res, df_long=dfl, region_subset=region_subset, perm_fits=perms),
        "group×segment×region" : omnibus_group_x_segment_x_region(res, df_long=dfl, region_subset=region_subset, perm_fits=perms),
    }

    print(f"[{fband}] {scope_label} omnibus tests (cached permutations{' present' if perms else ' absent'})")
    for label, out in tests.items():
        line = (f"  {label:<24} χ²={out['chi2']:.6g}  "
                f"df={out['df'] if out['df'] is not None else 'NA':>3}  "
                f"k={out['k_constraints']:>2}  p_wald={out['p_wald']:.6g}")
        if perms is not None:
            line += f"  p_perm={out['p_perm']:.6g}  (n_perm_used={out['n_perm_used']})"
        print(line)

def omnibus_group_any_segment_within_region(res, df_long, region_label, perm_fits=None):
    """
    Perform a region-focused omnibus Wald test of group effects across segments.

    This function calculates two complementary statistical tests for a single brain region:
    
    1. **Interaction effect** (`out['interaction']`):
       Tests whether the between-group difference (e.g., B − A) varies across segments 
       relative to the reference segment. In other words, it tests the group × segment 
       interaction within the specified region. This excludes the main group effect at 
       the reference segment, isolating the temporal modulation of the group difference.
       
       - Null hypothesis (H0): The group difference is identical across all segments
         (relative to the reference segment) within the region.
       - Alternative hypothesis (H1): At least one segment differs from the reference in
         group effect.
       - Degrees of freedom: K−1, where K is the number of segments (reference segment
         is excluded).

    2. **Main effect** (`out['main']`):
       Tests whether there is a consistent overall group difference across all segments
       in the region. This includes the main group effect at the reference segment.
       
       - Null hypothesis (H0): The average group difference across segments in the region
         is zero.
       - Alternative hypothesis (H1): The average group difference is non-zero.
       - Degrees of freedom: 1 (contrast is averaged across segments).
       - Note: Currently implemented as a simple unweighted average; a variance-weighted
         average could be implemented for more precise estimation if segments have 
         differing variance.

    Parameters
    ----------
    res : GEEResults
        A fitted GEE (Generalized Estimating Equations) model result object. 
        Must contain coefficients for group, segment, region, and their interactions.
    df_long : pandas.DataFrame
        Long-format data frame used to fit `res`. Must contain columns for `segment` 
        and `region` to identify categories and reference levels.
    region_label : str
        The name of the brain region to test (e.g., 'frontal', 'parietal'). Must exist
        in the `region` factor of `df_long`.
    perm_fits : list, optional
        A list of permuted GEE results for permutation-based p-value calculation. Default
        is None.

    Returns
    -------
    out : dict
        A dictionary containing two entries:
        - 'interaction': dict returned by `_wald_from_L_with_cached_perm` for the group ×
          segment interaction within the region.
        - 'main': dict returned by `_wald_from_L_with_cached_perm` for the average main
          group effect across all segments in the region.
        
        Each dict has keys:
        {'p_wald', 'chi2', 'df', 'k_constraints', 'names', 'p_perm', 'n_perm_used'}.
        If no valid contrast exists (e.g., missing segments), values are NaN and df=None.

    Notes
    -----
    - The function relies on helper functions:
        - `build_L_region_any_segment` to construct per-segment group-contrast rows.
        - `_wald_from_L_with_cached_perm` to compute Wald statistics, optionally using
          permutation-based inference.
        - `_get_segment_region_categories` to determine segment and region categories 
          and reference levels.
    - For the interaction test, the reference segment is subtracted from all other
      segments to isolate temporal differences in group effect.
    - For the main effect, all segment rows are averaged to produce a single 1-df contrast
      representing the overall group difference in the region.
    - Missing segments or regions in the fitted model are skipped; if no valid contrasts
      remain, the function returns NaNs.
    """
    if df_long is None:
        raise ValueError("df_long is required for region-focused omnibus tests.")

    out = {}

    # build per-segment L matrix for this region
    L_all = build_L_region_any_segment(res, df_long, region_label)
    if L_all is None or L_all.size == 0:
        empty = {'p_wald': np.nan, 'chi2': np.nan, 'df': None,
                'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0}
        out['interaction'] = empty.copy()
        out['main'] = empty.copy()
        return out

    # Map segments to row indices safely
    seg_cats, _, seg_ref, _ = _get_segment_region_categories(df_long)
    seg_to_idx = {str(seg): i for i, seg in enumerate(seg_cats)}
    if str(seg_ref) not in seg_to_idx:
        raise ValueError(f"Reference segment '{seg_ref}' not found in L_all rows")
    
    # Extract reference segment row
    ref_idx = seg_to_idx[str(seg_ref)]

    # --- Interaction (interaction-only, exclude main at reference) ---
    # Check that the reference row exists in L_all
    if ref_idx < 0 or ref_idx >= L_all.shape[0]:
        # fallback: cannot compute interaction contrasts safely
        out['interaction'] = {'p_wald': np.nan, 'chi2': np.nan, 'df': None,
                              'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0}
    else:
        L_ref = L_all[ref_idx:ref_idx+1, :]              # 1 x p
        # Remove reference row and subtract it from other rows to isolate interaction
        try:
            L_interaction = np.delete(L_all, ref_idx, axis=0) - L_ref  # (K-1) x p
        except Exception:
            L_interaction = None

        if L_interaction is None or L_interaction.size == 0:
            out['interaction'] = {'p_wald': np.nan, 'chi2': np.nan, 'df': None,
                                  'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0}
        else:
            res_inter = _wald_from_L_with_cached_perm(res, L_interaction, perm_fits=perm_fits)
            res_inter['names'] = []   # optionally could add per-seg labels
            out['interaction'] = res_inter

    # --- Main (average main group effect across all segments in this region) ---
    # Simple average across rows of L_all (each row is group effect at that segment)
    try:
        L_main = L_all.mean(axis=0, keepdims=True)  # 1 x p
    except Exception:
        L_main = None

    if L_main is None or L_main.size == 0:
        out['main'] = {'p_wald': np.nan, 'chi2': np.nan, 'df': None,
                       'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0}
    else:
        res_main = _wald_from_L_with_cached_perm(res, L_main, perm_fits=perm_fits)
        res_main['names'] = []
        out['main'] = res_main

    return out

def omnibus_group_any_region_within_segment(res, df_long, seg_label, region_subset=None, perm_fits=None):
    """
    Perform a segment-focused omnibus Wald test of group effects across regions.

    This function calculates two complementary statistical tests for a single time segment:

    1. **Interaction effect** (`out['interaction']`):
       Tests whether the between-group difference (e.g., B − A) varies across regions 
       relative to the reference region. In other words, it tests the group × region 
       interaction within the specified segment. This excludes the main group effect at 
       the reference region, isolating the spatial modulation of the group difference.
       
       - Null hypothesis (H0): The group difference is identical across all regions
         (relative to the reference region) within the segment.
       - Alternative hypothesis (H1): At least one region differs from the reference in
         group effect.
       - Degrees of freedom: R−1, where R is the number of regions (reference region
         is excluded).

    2. **Main effect** (`out['main']`):
       Tests whether there is a consistent overall group difference across all regions
       in the segment. This includes the main group effect at the reference region.
       
       - Null hypothesis (H0): The average group difference across regions in the segment
         is zero.
       - Alternative hypothesis (H1): The average group difference is non-zero.
       - Degrees of freedom: 1 (contrast is averaged across regions).
       - Note: Currently implemented as a simple unweighted average; a variance-weighted
         average could be implemented for more precise estimation if regions have 
         differing variance.

    Parameters
    ----------
    res : GEEResults
        A fitted GEE model result object with coefficients for group, region, segment, 
        and their interactions.
    df_long : pandas.DataFrame
        Long-format dataframe used to fit `res`. Must contain columns for `segment` 
        and `region` to identify categories and reference levels.
    seg_label : str
        The segment to test (e.g., 'E0', 'E1').
    region_subset : list of str, optional
        Specific regions to include in the test. Default is None (all regions).
    perm_fits : list, optional
        A list of permuted GEE results for permutation-based p-value calculation. Default is None.

    Returns
    -------
    out : dict
        A dictionary containing two entries:
        - 'interaction': dict returned by `_wald_from_L_with_cached_perm` for the group ×
          region interaction within the segment.
        - 'main': dict returned by `_wald_from_L_with_cached_perm` for the average main
          group effect across all regions in the segment.

        Each dict has keys:
        {'p_wald', 'chi2', 'df', 'k_constraints', 'names', 'p_perm', 'n_perm_used'}.
        If no valid contrast exists (e.g., missing regions), values are NaN and df=None.

    Notes
    -----
    - Relies on helper functions:
        - `build_L_segment_any_region` to construct per-region group-contrast rows.
        - `_wald_from_L_with_cached_perm` to compute Wald statistics.
        - `_get_segment_region_categories` to determine segment and region categories 
          and reference levels.
    - Builds L matrices for all regions in the segment to ensure reference region is included.
    - For the interaction test, the reference region is subtracted from all other
      regions to isolate spatial differences in group effect.
    - For the main effect, all region rows are averaged to produce a single 1-df contrast
      representing the overall group difference in the segment.
    - Missing regions or reference region in the fitted model are skipped.
    """
    if df_long is None:
        raise ValueError("df_long is required for region-focused omnibus tests.")
    
    # Get segment and region categories
    seg_cats, region_cats, seg_ref, region_ref = _get_segment_region_categories(df_long)
    seg_cats_str = list(map(str, seg_cats))
    seg_label_str = str(seg_label)
    if seg_label_str not in seg_cats_str:
        raise ValueError(f"Time segment '{seg_label}' not found in data. Available: {seg_cats_str}")

    region_cats_str = list(map(str, region_cats))
    if region_subset is None or region_subset == 'all':
        region_subset = region_cats_str
    else:
        region_subset = [str(r) for r in region_subset]

    out = {}

    # Build per-region L matrix for this segment (optionally subset regions)
    L_all = build_L_segment_any_region(res, df_long, seg_label, region_subset=None)
    if L_all is None or L_all.size == 0:
        empty = {'p_wald': np.nan, 'chi2': np.nan, 'df': None,
                 'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0}
        out['interaction'] = empty.copy()
        out['main'] = empty.copy()
        return out

    # Extract reference region row
    if str(region_ref) not in region_cats_str:
        raise ValueError(f"Reference region '{region_ref}' not found in data")
    ref_idx = region_cats_str.index(str(region_ref))
    L_ref = L_all[ref_idx:ref_idx+1, :]  # 1 x p

    # --- Interaction (exclude reference region, isolate group x region) ---
    try:
        # Only include rows for region_subset excluding reference
        subset_idx = [i for i, r in enumerate(region_cats_str) if r in region_subset and r != str(region_ref)]
        if subset_idx:
            L_subset = L_all[subset_idx, :]
            L_interaction = L_subset - L_ref
            out['interaction'] = _wald_from_L_with_cached_perm(res, L_interaction, perm_fits=perm_fits)     
        else:
            out['interaction'] = {'p_wald': np.nan, 'chi2': np.nan, 'df': None,
                                  'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0}
    except Exception:
        out['interaction'] = {'p_wald': np.nan, 'chi2': np.nan, 'df': None,
                              'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0}


    # --- Main effect (average group effect across regions, include reference if in subset) ---
    try:
        subset_idx_all = [i for i, r in enumerate(region_cats_str) if r in region_subset]
        if subset_idx_all:
            L_main_subset = L_all[subset_idx_all, :]
            L_main_avg = L_main_subset.mean(axis=0, keepdims=True)  # 1 x p
            out['main'] = _wald_from_L_with_cached_perm(res, L_main_avg, perm_fits=perm_fits)
            out['main']['names'] = []
        else:
            out['main'] = {'p_wald': np.nan, 'chi2': np.nan, 'df': None,
                           'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0}
    except Exception:
        out['main'] = {'p_wald': np.nan, 'chi2': np.nan, 'df': None,
                       'k_constraints': 0, 'names': [], 'p_perm': np.nan, 'n_perm_used': 0}

    return out

# -------- convenience printer across all regions ------------------
def print_region_any_segment_summary(res_by_band, df_long_by_band, perm_fits_by_band, fband, region_subset=None):
    """
    Print region-focused omnibus tests: 'any segment' B−A within each region.
    
    Parameters:
    - res_by_band: dict like {'alpha': gee_result, ...}
    - df_long_by_band: dict like {'alpha': df_long, ...}
    - perm_fits_by_band: dict like {'alpha': [gee_result_perm1, ...], ...} or None
    - fband: frequency band name (key for the above dicts)
    - region_subset: list of region names to test, or None for all regions
    """
    res = res_by_band[fband]
    dfl = df_long_by_band[fband]
    perm_fits = None if perm_fits_by_band is None else perm_fits_by_band.get(fband)

    if dfl is None:
        raise ValueError("df_long is required for region-focused omnibus tests.")

    _, region_cats, _, _ = _get_segment_region_categories(dfl)
    region_cats_str = list(map(str, region_cats))

    # Determine which regions to test
    if region_subset is not None:
        region_subset_str = [str(r) for r in region_subset]
        # Validate that all requested regions exist
        missing_regions = [r for r in region_subset_str if r not in region_cats_str]
        if missing_regions:
            raise ValueError(f"Regions not found in data: {missing_regions}. Available: {region_cats_str}")
        region_list = region_subset_str
        scope_label = f"Region-subset {region_subset} focused"
    else:
        region_list = region_cats_str
        scope_label = "All regions"

    print(f"[{fband}] {scope_label} omnibus: 'any segment' B−A within region")
    
    for reg in region_list:
        out_all = omnibus_group_any_segment_within_region(res, dfl, region_label=reg, perm_fits=perm_fits)
        
        for effect_type, out in out_all.items():
            chi2 = out['chi2']
            df = out['df']
            k = out['k_constraints']
            line = (f"{effect_type:<11} region={reg:<12} χ²={chi2:.6g}  df={df if df is not None else 'NA':>3}  "
                    f"k={k:>2}  p_wald={out['p_wald']:.6g}")
            if np.isfinite(out['p_perm']):
                line += f"  p_perm={out['p_perm']:.6g}  (n_perm_used={out['n_perm_used']})"
            print(line)

def print_segment_any_region_summary(res_by_band, df_long_by_band, perm_fits_by_band, fband, region_subset=None):
    """
    Print segment-focused omnibus tests:'any region' B-A within each segment.
    """
    res = res_by_band[fband]
    dfl = df_long_by_band[fband]
    perm_fits = None if perm_fits_by_band is None else perm_fits_by_band.get(fband)

    if dfl is None:
        raise ValueError("df_long is required for region-focused omnibus tests.")

    seg_cats, _, _, _ = _get_segment_region_categories(dfl)
    seg_cats_str = list(map(str, seg_cats))

    for segment in seg_cats_str:
        out_all = omnibus_group_any_region_within_segment(res, dfl, seg_label=segment, region_subset=region_subset, perm_fits=perm_fits)

        for effect_type, out in out_all.items():
            chi2 = out['chi2']
            df = out['df']
            k = out['k_constraints']
            line = (f"{effect_type:<11} segment={segment:<12} χ²={chi2:.6g}  df={df if df is not None else 'NA':>3}  "
                    f"k={k:>2}  p_wald={out['p_wald']:.6g}")
            if np.isfinite(out['p_perm']):
                line += f"  p_perm={out['p_perm']:.6g}  (n_perm_used={out['n_perm_used']})"
            print(line)

# ---------------------------------------------------------------------------------------------
# build omnibus table for a single band
# ----------------------------------------------------------------------------------------------
def _omnibus_summary_for_band(
    fband: str,
    res,                # GEEResults for this band
    dfl: pd.DataFrame,  # df_long for this band (needed for region-focused omnibus)
    perm_fits,          # list[GEEResults] or None
    *,
    include_global: bool = True,
    include_region_focused: bool = True,
    include_time_focused: bool = True,
    region_subset: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a tidy omnibus table for one band using already-computed perm_fits.
    Returns columns:
      ['fband','scope','test','region','chi2','df','k_constraints','p_wald','p_perm','n_perm_used']
    """
    rows = []

    def _row(scope, test, out, region_epoch=None):
        rows.append({
            "fband": fband,
            "scope": scope,                 # 'global', 'region_subset', 'region_focused'
            "test":  test,                  # e.g., 'ANY group term', 'group×segment', ...
            "region/epoch": region_epoch,               # None for global rows
            "chi2": out.get("chi2", np.nan),
            "df": out.get("df", None),
            "k_constraints": out.get("k_constraints", 0),
            "p_wald": out.get("p_wald", np.nan),
            "p_perm": out.get("p_perm", np.nan),
            "n_perm_used": out.get("n_perm_used", 0),
        })

    # Validate region_subset if provided
    if region_subset is not None:
        if dfl is None:
            raise ValueError("df_long is required when region_subset is provided.")
        _, region_cats, _, _ = _get_segment_region_categories(dfl)
        region_cats_str = list(map(str, region_cats))
        missing_regions = [r for r in map(str, region_subset) if r not in region_cats_str]
        if missing_regions:
            raise ValueError(f"Regions not found in data: {missing_regions}. Available: {region_cats_str}")

    # Global or region-subset omnibus tests
    if include_global:
        scope_label = "global" if region_subset is None else "region_subset"
        # Pass df_long and region_subset to all omnibus functions
        _row(scope_label, "ANY group term", 
             omnibus_any_group(res, df_long=dfl, region_subset=region_subset, perm_fits=perm_fits))
        _row(scope_label, "group×segment", 
             omnibus_group_x_segment(res, df_long=dfl, region_subset=region_subset, perm_fits=perm_fits))
        _row(scope_label, "group×region", 
             omnibus_group_x_region(res, df_long=dfl, region_subset=region_subset, perm_fits=perm_fits))
        _row(scope_label, "group×segment×region", 
             omnibus_group_x_segment_x_region(res, df_long=dfl, region_subset=region_subset, perm_fits=perm_fits))

    # Region-focused omnibus: within-region “any segment” B−A ≠ 0?
    if include_region_focused:
        if dfl is None:
            raise ValueError("df_long is required for region-focused omnibus tests.")
        _, region_cats, _, _ = _get_segment_region_categories(dfl)
        region_list = [str(r) for r in region_cats]
        # if region_subset is not None:
        #     wanted = set(map(str, region_subset))
        #     region_list = [r for r in region_list if r in wanted]
        for reg in region_list:
            out_all = omnibus_group_any_segment_within_region(res, dfl, region_label=reg, perm_fits=perm_fits)
            for effect_type, out in out_all.items():
                _row("region_focused", effect_type, out, region_epoch=reg)

    if include_time_focused:
        if dfl is None:
            raise ValueError("df_long is required for region-focused omnibus tests.")
        seg_cats, _, _, _ = _get_segment_region_categories(dfl)
        seg_cats_str = [str(s) for s in seg_cats]
        for seg in seg_cats_str:
            out_all = omnibus_group_any_region_within_segment(res, dfl, seg_label=seg, region_subset=region_subset, perm_fits=perm_fits)
            for effect_type, out in out_all.items():
                _row("time_focused", effect_type, out, region_epoch=seg)

    omni = pd.DataFrame(rows)
    if not omni.empty:
        omni["scope"] = pd.Categorical(omni["scope"], 
                                    categories=["global", "region_subset", "region_focused", "time_focused"],
                                    ordered=True)
        omni = omni.sort_values(["scope","test","region/epoch"], kind="mergesort").reset_index(drop=True)
    return omni

# ---------------------------------------------------------------------------------------------------
# Cellwise group differences (B−A) with CIs at each (segment, region) for a band
# ----------------------------------------------------------------------------------------------------
def group_diff_cell(
    res, 
    seg_label, 
    region_label, 
    seg_ref, 
    region_ref, 
    *,
    alpha=0.05,
    perm_fits=None,
):
    """
    Linear contrast for the group effect (B−A) at a given (segment, region).
    Reference coding: group='A', segment=seg_ref, region=region_ref.

    Contrast sums:
      C(group)[T.B]
      + C(group)[T.B]:C(segment)[T.seg_label]                        (if seg != seg_ref)
      + C(group)[T.B]:C(region)[T.region_label]                      (if region != region_ref)
      + C(group)[T.B]:C(segment)[T.seg_label]:C(region)[T.region_label] (if both != ref)

    Returns:
      estimate, se, z, p          (Wald, χ²(1))
      ci_lo, ci_hi                (normal-approx (1−alpha) CI)
      wald_stat                   (χ²(1) Wald statistic for this contrast)
      p_perm, perm_stat_obs, n_perm_used   (only if n_perm>0 and df_long provided)
    """
    # Build contrast on observed fit
    keep = _cell_keep_selector(res, seg_label, region_label, seg_ref, region_ref)

    # Build 1×p contrast on the observed fit
    L, kept_names, _ = _build_L_from_selector(res, keep)
    if L is None or L.size == 0:
        raise ValueError(f"No coefficients matched for seg='{seg_label}', region='{region_label}'. "
                         "Check category names and references.")

    L = L.sum(axis=0, keepdims=True)  # shape (1, p)

    # Wald estimate/SE/p form the fitted model 
    beta = res.params.to_numpy(dtype=float)
    V    = res.cov_params().to_numpy(dtype=float)

    est = float((L @ beta).item())
    var = float((L @ V @ L.T).item())
    var = max(var, 0.0) 
    se  = np.sqrt(var) if var >= 0 else np.nan
    z   = est / se if se > 0 else np.nan

    wald = res.wald_test(L, scalar=True)
    p_wald = float(np.asarray(wald.pvalue).reshape(()))
    wald_stat = float(np.asarray(wald.statistic).reshape(()))

    # two-sided (1 - alpha) CI using normal critical value
    zcrit = norm.ppf(1 - alpha/2.0)
    ci_lo = est - zcrit*se if np.isfinite(se) else np.nan
    ci_hi = est + zcrit*se if np.isfinite(se) else np.nan

    out =  {
        "estimate": est, "se": se, "z": float(z),
        "p_wald": p_wald, "wald_stat": wald_stat,
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)
    }

    # Optional permutation p-value
    if perm_fits is not None:

        obs_stat = wald_stat
        ge_count = 0
        n_used = 0

        for rp in perm_fits:
            obs_names = res.params.index
            rp_names  = rp.params.index
            if not rp_names.equals(obs_names):
                # Reorder L's columns from OBS order into RP order
                col_idx = [obs_names.get_loc(nm) for nm in rp_names]
                Lp = L[:, col_idx]
            else:
                Lp = L
            
            wtest_p = rp.wald_test(Lp, scalar=True) # << NOTE: reuse the same L
            stat_p = float(np.asarray(wtest_p.statistic).reshape(()))
            if not np.isfinite(stat_p):
                continue
            n_used += 1
            if stat_p >= obs_stat - 1e-12:
                ge_count += 1

        if n_used == 0:
            p_perm = np.nan
        else:
            # add-one smoothing
            p_perm = (ge_count + 1) / (n_used + 1)

        out.update({
            "p_perm": float(p_perm) if np.isfinite(p_perm) else np.nan,
            "perm_stat_obs": obs_stat,
            "n_perm_used": int(n_used)
        })

    return out

def cellwise_group_table(
    res, 
    df_long,  
    *,
    alpha=0.05,
    perm_fits,
    fdr=True,
    region_subset=None
):
    """
    Build a table of group (B−A) contrasts at every (segment, region) cell,
    with normal-approx CIs, Wald p, and optional permutation p.
    BH-FDR and Holm can be applied separately to Wald and permutation p-values.

    The reference levels are the first category of 'segment' and 'region'.
    """

    # categories & references
    seg_cats, region_cats, seg_ref, region_ref = _get_segment_region_categories(df_long)
    if region_subset is None:
        region_subset = region_cats

    rows = []
    for seg in seg_cats:
        for reg in region_subset:
            out = group_diff_cell(
                res, 
                str(seg), str(reg), 
                seg_ref, region_ref,
                alpha=alpha,
                perm_fits=perm_fits
            )
            rows.append({"segment": seg, "region": reg, **out})

    tab = pd.DataFrame(rows)

    # human-friendly labels (E1..E5 assuming 0-based coding)
    try:
        tab['E'] = tab['segment'].astype(int).map(lambda s: f"E{s+1}")
    except Exception:
        pass

    if fdr:
        mask = tab['p_wald'].notna().to_numpy()
        tab = tab.copy()
        tab['p_fdr'] = np.nan
        tab['p_holm'] = np.nan
        if mask.sum() > 0:
            tab.loc[mask, 'p_fdr'] = multipletests(tab.loc[mask, 'p_wald'], alpha=alpha, method='fdr_bh')[1]
            tab.loc[mask, 'p_holm'] = multipletests(tab.loc[mask, 'p_wald'], alpha=alpha, method='holm')[1]

        if 'p_perm' in tab.columns:
            mask_perm = tab['p_perm'].notna().to_numpy()
            if mask_perm.sum() > 0:
                tab['p_perm_fdr']  = np.nan
                tab['p_perm_holm'] = np.nan
                tab.loc[mask_perm, 'p_perm_fdr']  = multipletests(tab.loc[mask_perm, 'p_perm'], alpha=alpha, method='fdr_bh')[1]
                tab.loc[mask_perm, 'p_perm_holm'] = multipletests(tab.loc[mask_perm, 'p_perm'], alpha=alpha, method='holm')[1]


    # order columns for readability
    col_order = ['region', 'E', 'segment', 
                 'estimate', 'se', 'ci_lo', 'ci_hi', 
                 'p_wald', 'wald_stat']
    if 'p_fdr' in tab.columns:  col_order.append('p_fdr')
    if 'p_holm' in tab.columns: col_order.append('p_holm')
    if 'p_perm' in tab.columns:
        col_order += ['p_perm', 'n_perm_used']
        if 'p_perm_fdr' in tab.columns:  col_order.append('p_perm_fdr')
        if 'p_perm_holm' in tab.columns: col_order.append('p_perm_holm')

    return tab[[c for c in col_order if c in tab.columns]]

def _seed_for_band(base_seed: int | None, fband: str) -> int | None:
    """Stable per-band seed (reproducible across runs)."""
    if base_seed is None:
        return None
    return (zlib.adler32(fband.encode('utf-8')) ^ int(base_seed)) & 0xFFFFFFFF


def build_gee_omnibus_and_cellwise_tables(
    gee_by_band: dict,
    df_long_by_band: dict,
    formula: str | None = None,
    *,
    n_perm_per_band: int = 1000,          # set 0 to skip perms (Wald-only)
    base_seed: int | None = 42,
    bands_order: list[str] | None = None, # display order; defaults to keys order
    region_order: list[str] | None = None,
    alpha: float = 0.05,
    fdr: bool = True,
    verbose: bool = True,
    print_omnibus: bool = False,
    include_global_omnibus: bool = True,
    include_region_focused_omnibus: bool = False,
    include_time_focused_omnibus:bool = False,
    region_subset: list[str] | None = None,
    return_omnibus_df: bool = True
) -> pd.DataFrame:
    """
    Run cell-wise tables per band (with optional permutations) and, in the same
    loop where `perm_fits` are available, also build an omnibus Wald table per band.
    Returns:
      - if return_omnibus_df=False: cellwise DF (as before)
      - if return_omnibus_df=True : (cellwise DF, omnibus DF)
    """
    if formula is None:
        formula = "y ~ C(group) * C(segment) * C(region)"

    if bands_order is None:
        bands_order = list(gee_by_band.keys())
    if region_order is None:
        region_order = ['prefrontal', 'frontal', 'central', 'parietal']  # default

    all_tabs = []
    omni_tabs = [] 

    for fband in bands_order:
        if verbose:
            print(f"\n=== Processing band: {fband} ===")
        res = gee_by_band[fband]
        dfl = df_long_by_band[fband]

        # Build cached permuted fits for THIS band (or None to skip)
        if n_perm_per_band and n_perm_per_band > 0:
            seed = _seed_for_band(base_seed, fband)
            perm_fits, meta = prefit_permuted_gees_for_band(
                formula=formula, df_long=dfl, n_perm=n_perm_per_band, seed=seed, verbose=verbose
            )
            if verbose:
                print(f"  Perm-fits used: {meta['n_used']} (tries={meta['n_tried']})")
            # perm_dict_for_band = {fband: perm_fits}
        else:
            perm_fits = None
            # perm_dict_for_band = None
            if verbose:
                print("  Skipping permutations (Wald-only).")

        # ---- (optional) PRINT omnibus summaries
        if print_omnibus:
            # General omnibus (ANY group term, G×seg, G×reg, G×seg×reg)
            print_omnibus_summary(
                res_by_band= gee_by_band, 
                fband=fband, 
                df_long_by_band = df_long_by_band,
                perm_fits_by_band={fband: perm_fits},
                region_subset=region_subset,
            )

            # Region-focused omnibus: within each region, “any segment” B−A ≠ 0?
            print_region_any_segment_summary(
                res_by_band=gee_by_band,
                df_long_by_band=df_long_by_band,
                perm_fits_by_band={fband: perm_fits},
                fband=fband,
                # region_subset=region_subset
            )

            # Time(segment)-focused omnibus: within each segment, "any region" B-A ≠ 0?
            print_segment_any_region_summary(
                res_by_band=gee_by_band,
                df_long_by_band=df_long_by_band,
                perm_fits_by_band={fband: perm_fits},
                fband=fband,
                # region_subset=region_subset
            )

        omni_tab = _omnibus_summary_for_band(
            fband, res, dfl, perm_fits,
            include_global=include_global_omnibus,
            include_region_focused=include_region_focused_omnibus,
            include_time_focused=include_time_focused_omnibus,
            region_subset=region_subset,
        )

        if not omni_tab.empty:
            omni_tabs.append(omni_tab)

        # -------- Cellwise table for this band
        tab = cellwise_group_table(
            res,
            dfl,
            alpha=alpha,
            perm_fits=perm_fits,   # None => no perm p-values; list => adds p_perm
            fdr=fdr,
            # region_subset=region_subset
        ).copy()

        # Annotate and prep ordering
        tab["fband"] = fband
        if "segment_idx" not in tab.columns:
            # try numeric; fall back to extracting the integer from labels like "E3"
            seg_num = pd.to_numeric(tab["segment"], errors="coerce")
            if seg_num.isna().any() and "E" in str(tab["segment"].iloc[0]):
                seg_num = tab["segment"].astype(str).str.extract(r'(\d+)').astype(float)[0]
            tab["segment_idx"] = seg_num        
        
        all_tabs.append(tab)

        # Free heavy objects before next band
        del perm_fits
        gc.collect()

    # ----------------------------------------------------
    # Concatenate & sort
    # ----------------------------------------------------
    
    # Cell-wise
    df_cell_wise = pd.concat(all_tabs, ignore_index=True)
    if not df_cell_wise.empty:
        df_cell_wise["fband"] = pd.Categorical(df_cell_wise["fband"], categories=bands_order, ordered=True)
        
        # region custom order: keep desired order first, append any others (if present)
        unique_regions = df_cell_wise["region"].astype(str).unique().tolist()
        present = [r for r in region_order if r in unique_regions]
        others  = [r for r in unique_regions if r not in present]
        region_cats = present + sorted(others)
        df_cell_wise["region"] = pd.Categorical(df_cell_wise["region"], categories=region_cats, ordered=True)

        # final sort: fband → region → segment
        df_cell_wise = df_cell_wise.sort_values(["fband", "region", "segment_idx"], kind="mergesort").reset_index(drop=True)

        # column layout
        preferred_cols = [
            "fband", "region", 
            # "E",
            "segment", 
            # "segment_idx",
            "estimate", 
            # "se", 
            "ci_lo", "ci_hi",
            "p_wald", 
            # "wald_stat",
            "p_fdr", 
            # "p_holm",
            "p_perm", 
            # "n_perm_used", 
            "p_perm_fdr", 
            # "p_perm_holm",
        ]
        df_cell_wise = df_cell_wise[[c for c in preferred_cols if c in df_cell_wise.columns]]
    
    # Omnibus
    if omni_tabs:
        omni_all = pd.concat(omni_tabs, ignore_index=True)
        omni_all["fband"] = pd.Categorical(omni_all["fband"], categories=bands_order, ordered=True)
        omni_all = omni_all.sort_values(["fband","scope","test","region/epoch"], kind="mergesort").reset_index(drop=True)
    else:
        omni_all = pd.DataFrame(columns=[
            "fband","scope","test","region/epoch","chi2","df",
            # "k_constraints",
            "p_wald","p_perm",
            # "n_perm_used"
        ])

    return (df_cell_wise, omni_all) if return_omnibus_df else df_cell_wise

#=====================================================================================================================
# CAN BE ADDED LATER FOR OMNIBUS TESTS WITH LOWER DF  
#=====================================================================================================================
# 1) Linear trend (df=1) averaged over a region subset
def _L_group_linear_trend(res, df_long, regions_subset):
    seg_cats, region_cats, seg_ref, region_ref = _get_segment_region_categories(df_long)
    # centered numeric scores for ordered segments (works for K up to 10)
    scores = {s: i for i, s in enumerate(seg_cats)}
    mu = np.mean(list(scores.values()))
    scores = {s: (v - mu) for s, v in scores.items()}
    rows = []
    for r in regions_subset:
        row = np.zeros(len(res.params))
        for s, w in scores.items():
            keep = _cell_keep_selector(res, str(s), str(r), seg_ref, region_ref)
            Ls, _, _ = _build_L_from_selector(res, keep)
            if Ls is not None and Ls.size:
                row += w * Ls.sum(axis=0)
        rows.append(row)
    L = np.mean(np.vstack(rows), axis=0, keepdims=True)  # 1×p
    return L

def gee_omnibus_linear_trend(res, df_long, regions_subset=('frontal','parietal'), perm_fits=None):
    L = _L_group_linear_trend(res, df_long, [str(r) for r in regions_subset])
    return _wald_from_L_with_cached_perm(res, L, perm_fits=perm_fits)

# 2) Difference-in-differences (E_last − E_first), df=1, averaged over subset
def _L_group_diff_in_diff(res, df_long, regions_subset, seg_lo=None, seg_hi=None):
    seg_cats, region_cats, seg_ref, region_ref = _get_segment_region_categories(df_long)
    seg_lo = seg_lo or seg_cats[0]
    seg_hi = seg_hi or seg_cats[-1]
    rows = []
    for r in regions_subset:
        keep_hi = _cell_keep_selector(res, str(seg_hi), str(r), seg_ref, region_ref)
        L_hi, _, _ = _build_L_from_selector(res, keep_hi); L_hi = L_hi.sum(axis=0, keepdims=True)
        keep_lo = _cell_keep_selector(res, str(seg_lo), str(r), seg_ref, region_ref)
        L_lo, _, _ = _build_L_from_selector(res, keep_lo); L_lo = L_lo.sum(axis=0, keepdims=True)
        rows.append(L_hi - L_lo)
    L = np.mean(np.vstack(rows), axis=0, keepdims=True)  # 1×p
    return L

def gee_omnibus_diff_in_diff(res, df_long, regions_subset=('frontal','parietal'), seg_lo=None, seg_hi=None, perm_fits=None):
    L = _L_group_diff_in_diff(res, df_long, [str(r) for r in regions_subset], seg_lo=seg_lo, seg_hi=seg_hi)
    return _wald_from_L_with_cached_perm(res, L, perm_fits=perm_fits)

# usage 
# subset = ['frontal','parietal']

# for fband in ['delta','theta','alpha','beta','gamma']:
#     res  = gee_by_band[fband]
#     dfl  = df_long_by_band[fband]
#     perms = perm_fits_by_band.get(fband) if 'perm_fits_by_band' in globals() else None

#     out_trend = gee_omnibus_linear_trend(res, dfl, regions_subset=subset, perm_fits=perms)     # df = 1
#     out_dod   = gee_omnibus_diff_in_diff(res, dfl, regions_subset=subset, perm_fits=perms)     # df = 1
#     out_pair  = omnibus_group_x_region(res, dfl, region_subset=subset, perm_fits=perms)        # df = K (your existing pairwise)

#     print(f"[{fband}] trend:    df={out_trend['k_constraints']}  p_wald={out_trend['p_wald']:.3g}  p_perm={out_trend['p_perm']:.3g}")
#     print(f"[{fband}] E_last-E1 df={out_dod['k_constraints']}    p_wald={out_dod['p_wald']:.3g}    p_perm={out_dod['p_perm']:.3g}")
#     print(f"[{fband}] par−fro   df={out_pair['k_constraints']}   p_wald={out_pair['p_wald']:.3g}   p_perm={out_pair['p_perm']:.3g}")

