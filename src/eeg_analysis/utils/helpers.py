import mne
import numpy as np
from scipy.signal import find_peaks

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
    Detects outliers in a time series using the IQR method and interpolates them using linear interpolation.
    
    Parameters:
    time_series (np.ndarray): The input time series data.
    
    Returns:
    np.ndarray: The time series with outliers interpolated.
    """
    if not isinstance(time_series, np.ndarray):
        raise ValueError("Input time series must be a numpy array")
    if time_series.ndim != 1:
        raise ValueError("Input time series must be one-dimensional")
    
    clean_series = time_series.astype(float)

    if len(time_series) >= 3:
        # Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = np.percentile(time_series, 25)
        Q3 = np.percentile(time_series, 75)
        
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
        
    return clean_series