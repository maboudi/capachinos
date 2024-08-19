from src.eeg_analysis.analysis.power_spectral import TimeFrequencyRepresentation
import numpy as np
import matplotlib.pyplot as plt

def plot_eeg(eeg, sample_rate, threshold, ax, start_time_offset=0, channel_labels=None, detect_avalanche=False, highlight_avalanche_period=False):
    """
    Plot EEG data with each channel on a separate row in a single Axes.
    
    Parameters:
    - eeg: NumPy array of shape (n_samples, n_channels)
    - sample_rate: Sampling rate of the EEG data
    - ax: Axes object where the EEG data is to be plotted
    - start_time_offset: Offset to start the time axis, allows for plotting multiple epochs sequentially
    - channel_labels: Optional list of strings with channel labels
    - detect_avalanche: Boolean flag to enable threshold line and over-threshold marking
    
    """
    n_samples, n_channels = eeg.shape
    time = np.arange(0, n_samples) / sample_rate + start_time_offset
    
    spacing = 6

    if channel_labels is None:
        channel_labels = [f'Channel {i+1}' for i in range(n_channels)]
    
    for i in range(n_channels):
        # Plot the signal trace
        ax.plot(time, eeg[:, i] + i * spacing, color='black')

        if detect_avalanche:
            # Add dashed lines
            ax.plot([time[0], time[-1]], [i * spacing - threshold, i * spacing - threshold], '--g', linewidth=0.5)
            ax.plot([time[0], time[-1]], [i * spacing + threshold, i * spacing + threshold], '--g', linewidth=0.5)
            
            # Highlight over-threshold parts
            above_threshold = np.where(eeg[:, i] > threshold)[0]
            below_threshold = np.where(eeg[:, i] < -threshold)[0]

            def plot_continuous(time, data, idx_ranges, color):
                for start_idx, end_idx in idx_ranges:
                    ax.plot(time[start_idx:end_idx+1], data[start_idx:end_idx+1] + i * spacing, color=color)

            if above_threshold.size > 0:
                above_diff = np.diff(above_threshold) > 1
                above_split_idx = np.where(above_diff)[0] + 1
                above_ranges = np.split(above_threshold, above_split_idx)
                above_ranges = [(r[0], r[-1]) for r in above_ranges if len(r) > 1]
                plot_continuous(time, eeg[:, i], above_ranges, color='red')

            # Find continuous ranges below the threshold
            if below_threshold.size > 0:
                below_diff = np.diff(below_threshold) > 1
                below_split_idx = np.where(below_diff)[0] + 1
                below_ranges = np.split(below_threshold, below_split_idx)
                below_ranges = [(r[0], r[-1]) for r in below_ranges if len(r) > 1]
                plot_continuous(time, eeg[:, i], below_ranges, color='red')

            # for idx in above_threshold:
            #     if idx > 0:
            #         ax.plot(time[idx-1:idx+1], eeg[idx-1:idx+1, i] + i * spacing, color='red')
            
            # for idx in below_threshold:
            #     if idx > 0:
            #         ax.plot(time[idx-1:idx+1], eeg[idx-1:idx+1, i] + i * spacing, color='red')
    
    if highlight_avalanche_period and detect_avalanche:
        ax.axvspan(time[0], time[-1], ymin=-3/(n_channels * spacing + 3), ymax=(n_channels * spacing + 3 - 3) / (n_channels * spacing + 3), facecolor='red', edgecolor='red', alpha=0.1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('EEG channel')
    ax.set_yticks([i * spacing for i in range(n_channels)])
    ax.set_yticklabels(channel_labels)
    ax.grid(True)

    plt.tight_layout()
    start_time_offset = time[-1]

    return start_time_offset

def plot_continuous_epochs(tfr, ax=None, **kwargs):

    if ax is None:
        raise ValueError("An axis must be provided when plotting multiple epochs")
        
    if isinstance(tfr, TimeFrequencyRepresentation):
        tfr = {'Continuous':tfr}

    cumulative_time_offset = 0.0  # Initialize cumulative time offset

    # Iterate over the sorted epochs
    for epoch_key, tfr in tfr.items():
        
        im = tfr.plot(ax=ax, start_time=cumulative_time_offset, **kwargs)
        
        # Add markers and annotations to indicate epochs
        start_time = cumulative_time_offset
        end_time = start_time + (tfr.times[-1] - tfr.times[0])
        ax.axvline(x=start_time, color='w', linestyle='--', linewidth=1)  # Start of epoch
        ax.axvline(x=end_time, color='w', linestyle='--', linewidth=1)  # End of epoch
        # ax.text((start_time + end_time) / 2, 50, epoch_key,
        #         horizontalalignment='center', verticalalignment='top',
        #         color='w', fontsize=5, clip_on=True)

        # Update the time offset for the next epoch
        cumulative_time_offset = end_time

    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Frequency (Hz)')
    fig = ax.get_figure()
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', label='Power (dB/Hz)', pad=0.01)
    cbar.outline.set_visible(False)
    ax.set_xlim(0, cumulative_time_offset)
