import mne
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from typing import Tuple, Optional

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