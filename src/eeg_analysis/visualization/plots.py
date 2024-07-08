from src.eeg_analysis.analysis.power_spectral import TimeFrequencyRepresentation

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
