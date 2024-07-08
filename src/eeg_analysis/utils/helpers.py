import mne

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