from src.eeg_analysis.analysis.power_spectral import TimeFrequencyRepresentation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from src.eeg_analysis.visualization.my_custom_style import set_custom_style
from src.eeg_analysis.utils.helpers import gini
import networkx as nx

BLUE = "#1f3b73"  # Dark Blue
RED = "#a82323"  # Dark Red

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

def plot_continuous_epochs(tfr, ax=None, colorbar=False, **kwargs):

    if ax is None:
        raise ValueError("An axis must be provided when plotting multiple epochs")
        
    if isinstance(tfr, TimeFrequencyRepresentation):
        tfr = {'Continuous':tfr}

    cumulative_time_offset = 0.0  # Initialize cumulative time offset

    # Iterate over the sorted epochs
    for epoch_key, epoch_tfr in tfr.items():
        
        im = epoch_tfr.plot(ax=ax, start_time=cumulative_time_offset, **kwargs)
        
        # Add markers and annotations to indicate epochs
        start_time = cumulative_time_offset
        end_time = start_time + (epoch_tfr.times[-1] - epoch_tfr.times[0])
        if len(tfr) > 1: # if there are more than one epoch
            ax.axvline(x=start_time, color='w', linestyle='--', linewidth=1)  # Start of epoch
            ax.axvline(x=end_time, color='w', linestyle='--', linewidth=1)  # End of epoch
        # ax.text((start_time + end_time) / 2, 50, epoch_key,
        #         horizontalalignment='center', verticalalignment='top',
        #         color='w', fontsize=5, clip_on=True)

        # Update the time offset for the next epoch
        cumulative_time_offset = end_time

    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Frequency (Hz)')
    
    if colorbar:
        # Create space on the right for the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label('Power (dB/Hz)', fontsize=8)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=5)

    ax.set_xlim(0, cumulative_time_offset)

    return im

def plot_wPLI_heatmaps(pli, epoch_name, vmin=0, vmax=1, max_freq=None, colormap='viridis', xlabel=None, title=None, title_color=None, ax=None):
    
    """
    Plots inter-region weighted Phase Locking Index (wPLI) heatmaps and dynamics over time.
    
    Args:
        pli: Object from the class PhaseLagIndex.
        epoch_name: Name of the epoch to analyze (e.g., 'emergence').
        vmin: Minimum value for the colormap.
        vmax: Maximum value for the colormap.
        max_freq: Maximum frequency to plot.
    """
    region_names = pli.region_names
    num_regions = len(region_names)
    region_acrs = generate_region_acronyms(region_names)

    win_centers = pli.win_centers[epoch_name]
    pairs = [(i, j) for i in range(num_regions) for j in range(i)]

    if max_freq is None:
        max_freq = pli.fmax
    frequency_mask = np.logical_and(pli.freqs >= 0, pli.freqs <= max_freq)


    plot_width, plot_height = 600, 230
    fontsize = 6

    set_custom_style() 
    if ax is None:
        fig = plt.figure(figsize=(plot_width / 72, plot_height / 72))
    else:
        fig = ax.figure

    if title is not None:
        fig.suptitle(title, x=0.05, ha='left', fontsize=fontsize, fontweight='bold', color = title_color if title_color is not None else 'k')

    # Heatmaps of wPLI
    gs = fig.add_gridspec(num_regions, num_regions, left=0.05, right=0.45, wspace=0.1, hspace=0.5)
    pli_win = pli.conn_wpli_win_anat[epoch_name]
    for r1, r2 in pairs:
        ax_sub = fig.add_subplot(gs[r1, r2])
        # change the following code to a code based on imshow
        curr_pli_win = pli_win[r1, r2, frequency_mask, :]
        im = ax_sub.imshow(
            gaussian_filter(curr_pli_win, sigma = 0), 
            aspect='auto', 
            origin='lower', 
            cmap=colormap, 
            vmin=vmin, vmax=vmax, 
            extent=[0, max(win_centers), pli.fmin, max_freq],
            interpolation='none')
        
        ax_sub.set_ylim([0, 35])
        ax_sub.set_title(f'{region_acrs[region_names[r1]]} - {region_acrs[region_names[r2]]}',
                         loc='left', fontsize=fontsize, pad=0.2)
        
        ax_sub.set_xticks([0, max(win_centers)])
        ax_sub.set_xticklabels([0, 1])

        if r1 <= 4:
            ax_sub.set_xticklabels([])
        else:
            xticklabels_old = ax_sub.get_xticklabels()
            xticklabels = np.full(len(xticklabels_old), '', dtype=object)
            xticklabels[0] = xticklabels_old[0].get_text()
            xticklabels[-1] = xticklabels_old[-1].get_text()
            ax_sub.set_xticklabels(xticklabels)
            
        ax_sub.set_yticks(np.arange(0, max_freq, 10))

        if r2 != 0:
            ax_sub.set_yticklabels([])

        if (r1 == 3) & (r2 == 0):
            ax_sub.set_ylabel('Frequency (Hz)')

        if xlabel is not None:
            if (r1 == 5) & (r2 == 2):
                ax_sub.set_xlabel(xlabel)
    
    # Add a colorbar
    cax = fig.add_subplot(gs[1, num_regions-1])
    cbar = plt.colorbar(im, cax=cax, orientation='vertical', shrink=0.6)
    old_pos = cax.get_position()
    new_pos = [old_pos.x0, old_pos.y0, old_pos.width*0.25, old_pos.height*2]
    cax.set_position(new_pos)
    cbar.set_label('wPLI', fontsize=5)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=5)


    # Dynamics over time

    # Check if the favg attribute is available
    if not hasattr(pli, 'conn_wpli_win_favg_anat'):
        return
    
    frequency_bands = list(pli.conn_wpli_win_favg[epoch_name].keys())
    sequential_cmap = plt.cm.YlOrRd(np.linspace(0.35, 1, len(frequency_bands) - 1))
    color_map = {band: sequential_cmap[i] for i, band in enumerate(frequency_bands[:-1])}
    color_map['bb'] = 'gray'
    greek_fband_names = generate_greek_band_names(frequency_bands)
    
    gs_favg = fig.add_gridspec(num_regions, num_regions, left=0.50, right=0.90, wspace=0.1, hspace=0.5)
    for r1, r2 in pairs:
        ax_sub = fig.add_subplot(gs_favg[r1, r2])
        for fband in frequency_bands:
            pli_win_favg = pli.conn_wpli_win_favg_anat[epoch_name][fband]
            if fband == 'bb':
                label = f'{'Broadband'}\n ({pli.fmin_bb_avg}-{pli.fmax_bb_avg}Hz)'
            else:
                label = f'{greek_fband_names[fband]} ({pli.frequency_bands_of_interest[fband][0]}-{pli.frequency_bands_of_interest[fband][1]}Hz)'
            ax_sub.plot(win_centers, pli_win_favg[r1, r2], linewidth=0.5, label=label, color=color_map[fband], alpha=0.7)

        ax_sub.set_ylim([0, np.max([pli.conn_wpli_win_favg_anat[epoch_name][fb][:] for fb in frequency_bands])])
        ax_sub.set_title(f'{region_acrs[region_names[r1]]} - {region_acrs[region_names[r2]]}', loc='left', fontsize=fontsize, pad=0.2)

        ax_sub.set_xticks(np.arange(0, max(win_centers), 1000))
        if r1 <= 4:
            ax_sub.set_xticklabels([])
        else:
            xticklabels_old = ax_sub.get_xticklabels()
            xticklabels = np.full(len(xticklabels_old), '', dtype=object)
            xticklabels[0] = xticklabels_old[0].get_text()
            xticklabels[-1] = xticklabels_old[-1].get_text()
            ax_sub.set_xticklabels(xticklabels)

        if r2 != 0:
            ax_sub.set_yticklabels([])

        if (r1 == 3) & (r2 == 0):
            ax_sub.set_ylabel('Weighted PLI')
        
        if xlabel is not None:
            if (r1 == 5) & (r2 == 2):
                ax_sub.set_xlabel(xlabel)

        ax_sub.spines['top'].set_visible(False)
        ax_sub.spines['right'].set_visible(False)

    # Add legend
    ax_legend = fig.add_subplot(gs_favg[1, num_regions-1])
    legend_lines, legend_labels = ax_sub.get_legend_handles_labels()
    ax_legend.legend(legend_lines, legend_labels, loc='center', fontsize=fontsize)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    ax_legend.tick_params(bottom=False, left=False)

def plot_wPLI_frequency_profile(pli, vmin=None, vmax=None, ax=None, suptitle=None):
    """
    plot the connectivity versus frequency profile for eah pair of channels. wPLI matrix with one time point

    Args:
        pli: an array of shape (n_channels, n_channels, n_freqs) 
        vmin, vmax: To set the y-range manually 
        ax: Axes object to plot the data.

    Returns:
        None
    """
    region_names = pli.region_names
    num_regions = len(region_names)
    region_acrs = generate_region_acronyms(region_names)

    plot_width, plot_height = 800, 460
    fontsize = 12

    pairs = [(i, j) for i in range(num_regions) for j in range(i)]

    lower_iqr, upper_iqr = pli.conn_wpli_iqr

    vmax = np.nanmax(upper_iqr) if vmax is None else vmax
    if vmin is None:
        vmin = np.nanmin(lower_iqr) 
        vmin = np.min([vmin, 0])

    set_custom_style() 
    
    if ax is None:
        fig = plt.figure(figsize=(plot_width / 72, plot_height / 72))
    else:
        fig = ax.figure

    if suptitle is not None and ax is not None:
        ax.set_title(suptitle, x=0.5, ha='left', fontsize=fontsize, fontweight='bold')

    sub_gs = ax.get_subplotspec().subgridspec(num_regions-1, num_regions-1, wspace=0.3, hspace=0.5)
    for r1, r2 in pairs:
        ax_sub = fig.add_subplot(sub_gs[r1-1, r2])
        
        ax_sub.plot(pli.freqs, pli.conn_wpli_median[r1, r2, :], color='black', linewidth=1) 
        
        ax_sub.fill_between(pli.freqs, lower_iqr[r1, r2, :], upper_iqr[r1, r2, :], color='black', alpha=0.25, linewidth=0)
        
        ax_sub.set_title(f'{region_acrs[region_names[r1]]} - {region_acrs[region_names[r2]]}',
                         loc='left', fontsize=fontsize, pad=0.2)
        if (r1 == num_regions - 1) & (r2 == 2):
            ax_sub.set_xlabel('Frequency (Hz)', fontsize=fontsize)
        if (r1 == 3) & (r2 == 0):
            ax_sub.set_ylabel('wPLI', fontsize=fontsize)

        if r2 != 0:
            ax_sub.set_yticklabels([])
            
        ax_sub.set_xlim([0, 35])
        ax_sub.set_xticks(np.arange(0, 35, 10))

        if r1 < num_regions - 1:
            ax_sub.set_xticklabels([])
        # else:
        #     xticklabels_old = ax_sub.get_xticklabels()
        #     xticklabels = np.full(len(xticklabels_old), '', dtype=object)
        #     xticklabels[0] = xticklabels_old[0].get_text()
        #     xticklabels[-1] = xticklabels_old[-1].get_text()
        #     ax_sub.set_xticklabels(xticklabels)

        ax_sub.set_ylim([vmin, vmax])
        ax_sub.grid(True, which='both', linestyle='--', linewidth=.75)
        ax_sub.tick_params(axis='both', which='major', labelsize=10, length=2)


def plot_wPLI_frequency_profile_separate_groups(pli, vmin=None, vmax=None, ax=None, suptitle=None):
    """
    plot the connectivity versus frequency profile for each pair of channels. wPLI matrix with one time point

    Args:
        pli: an array of shape (n_channels, n_channels, n_freqs) 
        ax: Axes object to plot the data.

    Returns:
        None
    """
    region_names = pli.region_names
    num_regions = len(region_names)
    region_acrs = generate_region_acronyms(region_names)

    plot_width, plot_height = 800, 460
    fontsize = 12

    pairs = [(i, j) for i in range(num_regions) for j in range(i)]

    median_A = pli.conn_wpli_median_A
    median_B = pli.conn_wpli_median_B
    lower_iqr_A, upper_iqr_A = pli.conn_wpli_iqr_A
    lower_iqr_B, upper_iqr_B = pli.conn_wpli_iqr_B

    vmax = (np.nanmax([np.nanmax(upper_iqr_A) if upper_iqr_A is not None else float('-inf'),
                       np.nanmax(upper_iqr_B) if upper_iqr_B is not None else float('-inf')])
            if vmax is None else vmax)
    
    if vmin is None:
        vmin = (np.nanmin([np.nanmin(lower_iqr_A) if lower_iqr_A is not None else float('inf'),
                           np.nanmin(lower_iqr_B) if lower_iqr_B is not None else float('inf')])
                if vmin is None else vmin)
        vmin = np.min([vmin, 0])

    set_custom_style() 
    
    if ax is None:
        fig = plt.figure(figsize=(plot_width / 72, plot_height / 72))
    else:
        fig = ax.figure

    if suptitle is not None and ax is not None:
        ax.set_title(suptitle, x=0.5, ha='left', fontsize=fontsize, fontweight='bold')

    sub_gs = ax.get_subplotspec().subgridspec(num_regions-1, num_regions-1, wspace=0.3, hspace=0.5)
    for idx, (r1, r2) in enumerate(pairs):
        ax_sub = fig.add_subplot(sub_gs[r1-1, r2])
        
        if median_A is not None:
            ax_sub.plot(pli.freqs, median_A[r1, r2, :], color=BLUE, linewidth=1, label='Group A') 
            ax_sub.fill_between(pli.freqs, lower_iqr_A[r1, r2, :], upper_iqr_A[r1, r2, :], color=BLUE, alpha=0.25, linewidth=0)
            
        if median_B is not None:
            ax_sub.plot(pli.freqs, median_B[r1, r2, :], color=RED, linewidth=1, label='Group B') 
            ax_sub.fill_between(pli.freqs, lower_iqr_B[r1, r2, :], upper_iqr_B[r1, r2, :], color=RED, alpha=0.25, linewidth=0)
            
        ax_sub.set_title(f'{region_acrs[region_names[r1]]} - {region_acrs[region_names[r2]]}',
                         loc='left', fontsize=fontsize, pad=0.2)
        
        if (r1 == num_regions - 1) & (r2 == 2):
            ax_sub.set_xlabel('Frequency (Hz)', fontsize=fontsize)
        if (r1 == 3) & (r2 == 0):
            ax_sub.set_ylabel('wPLI', fontsize=fontsize)

        if r2 != 0:
            ax_sub.set_yticklabels([])
            
        ax_sub.set_xlim([0, 35])
        ax_sub.set_xticks(np.arange(0, 35, 10))

        if r1 < num_regions - 1:
            ax_sub.set_xticklabels([])

        ax_sub.set_ylim([vmin, vmax])
        ax_sub.grid(True, which='both', linestyle='--', linewidth=.75)
        ax_sub.tick_params(axis='both', which='major', labelsize=10, length=2)
        
    # Add legend
    ax_legend = fig.add_subplot([0.85, 0.7, 0.1, 0.1])
    legend_lines, legend_labels = ax_sub.get_legend_handles_labels()
    ax_legend.legend(legend_lines, legend_labels, loc='center', fontsize=fontsize)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    ax_legend.tick_params(bottom=False, left=False)


def plot_wPLI_matrix_over_time(pli, epoch_name, fbands, num_windows=5, colormap='viridis', vmax=0.3):
    """
    Plots wPLI matrices over time for multiple frequency bands during a specified epoch.

    Args:
        pli: Object containing wPLI data with attribute `.conn_wpli_win_favg_anat[epoch_name][fband]`.
        epoch_name: Name of the epoch (e.g., 'emergence').
        fbands: List of frequency bands to analyze (e.g., ['alpha', 'beta']).
        num_windows: Number of time windows to divide the data into (default: 5).

    Returns:
        Displays the plot with each frequency band in a separate row.
    """
    region_names = pli.region_names
    num_regions = len(region_names)
    region_acrs =  generate_region_acronyms(region_names) 
    
    fbands = fbands[:-1] if 'bb' in fbands else fbands
    greek_fband_names = generate_greek_band_names(fbands)


    # Prepare figure layout
    num_cols = num_windows
    num_rows = len(fbands)
    plot_width = num_cols * 65
    plot_height = num_rows * 65

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(plot_width / 72, plot_height / 72), squeeze=False)

    # # Main title
    # fig.suptitle(
    #     f'wPLI over {epoch_name.capitalize()} (split into {num_windows} time windows)',
    #     x=0.5, y=0.98, ha='center', fontsize=12, fontweight='bold'
    # )

    for i, fband in enumerate(fbands):
        curr_pli = pli.conn_wpli_win_favg_anat[epoch_name][fband]

        # Divide the time period into `num_windows` segments
        step = curr_pli.shape[2] // num_windows
        if step == 0:
            raise ValueError(f"Number of windows ({num_windows}) exceeds available time points ({curr_pli.shape[2]}).")

        for win in range(num_windows):
            ax = axs[i, win]
            start_idx = win * step
            end_idx = start_idx + step if win < num_windows - 1 else curr_pli.shape[2]

            # Compute the average matrix over the window
            avg_pli_matrix = np.mean(curr_pli[:, :, start_idx:end_idx], axis=2)

            # Plot the matrix
            im = ax.imshow(avg_pli_matrix, cmap=colormap, origin='upper', vmin=0, vmax=vmax)
            if i == 0:
                ax.set_title(f"E{win + 1}", fontsize=6)
            ax.set_xticks(range(num_regions))
            ax.set_yticks(range(num_regions))

            if win == 0:  # Add labels for the first column
                ax.set_yticklabels([region_acrs[name] for name in region_names], fontsize=6)
                ax.set_ylabel(f'{greek_fband_names[fband]}\n({pli.frequency_bands_of_interest[fband][0]}-{pli.frequency_bands_of_interest[fband][1]}Hz)', fontsize=6)
            else:
                ax.set_yticklabels([])

            if i == num_rows - 1:  # Add x-axis labels for the last row
                ax.set_xticklabels([region_acrs[name] for name in region_names], rotation=90, fontsize=6)
            else:
                ax.set_xticklabels([])

    # Add a colorbar
    cax = fig.add_subplot([0.7, 0.96, 0.1, 0.02])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal', shrink=0.6)
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label('wPLI', fontsize=5, labelpad=1)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=5)

    # Hide unused axes
    for i in range(num_rows):
        for j in range(num_windows, axs.shape[1]):
            axs[i, j].set_visible(False)

    # Adjust layout
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)

def plot_wPLI_metrics(pli, epoch_name, fbands):
    """
    Plots metrics for wPLI across time, including mean wPLI and Gini coefficient.

    Args:
        pli: Object containing wPLI data and metadata.
        epoch_name (str): Name of the epoch (e.g., 'emergence').
        fbands (list of str): List of frequency bands to analyze (e.g., ['alpha', 'beta']).
    Returns:
        None
    """
    # Frequency bands and colors
    frequency_bands = list(pli.conn_wpli_win_favg_anat[epoch_name].keys())
    sequential_cmap = plt.cm.YlOrRd(np.linspace(0.35, 1, len(frequency_bands) - 1))
    color_map = {band: sequential_cmap[i] for i, band in enumerate(frequency_bands[:-1])}
    color_map['bb'] = 'gray'

    fbands = fbands[:-1] if 'bb' in fbands else fbands
    greek_fband_names = generate_greek_band_names(fbands)

    # Plot setup
    plot_width = 300
    plot_height = 100

    set_custom_style() 
    fig, axs = plt.subplots(1, 3, figsize=(plot_width / 72, plot_height / 72), gridspec_kw={'width_ratios': [3, 3, 1]})
    fontsize = 6

    for fband in fbands:
        curr_pli_win = pli.conn_wpli_win_favg_anat[epoch_name][fband]
        num_win = curr_pli_win.shape[2]

        # Metrics storage
        network_connectedness = np.full((num_win,), np.nan)
        network_connect_std = np.full((num_win,), np.nan)
        gini_coeff = np.full((num_win,), np.nan)

        # Calculate metrics
        for win in range(num_win):
            pli_values = bottom_left_off_diagonal(curr_pli_win[:, :, win])
            network_connectedness[win] = np.nanmean(pli_values)
            network_connect_std[win] = np.std(pli_values)
            gini_coeff[win] = gini(pli_values)

        # Time window centers
        win_centers = pli.win_centers[epoch_name]
        color = color_map[fband]
        zorder = 0 if fband == 'bb' else 1

        # Band label
        if fband == 'bb':
            label = f"Broadband\n ({pli.fmin_bb_avg}-{pli.fmax_bb_avg}Hz)"
        else:
            label = f'{greek_fband_names[fband]}\n({pli.frequency_bands_of_interest[fband][0]}-{pli.frequency_bands_of_interest[fband][1]}Hz)'
        # Plot metrics
        axs[0].plot(win_centers, network_connectedness, linewidth=0.5, color=color, zorder=zorder, label=label, alpha=0.7)
        axs[1].plot(win_centers, gini_coeff, linewidth=0.5, color=color, zorder=zorder, label=label, alpha=0.7)

    # Customize axes
    axs[0].set_xticks(np.arange(0, max(win_centers), 1000))
    axs[0].set_xlabel(f"{epoch_name.capitalize()} time (s)", fontsize=fontsize)
    axs[0].set_ylabel("Mean wPLI between channels")

    axs[1].set_xticks(np.arange(0, max(win_centers), 1000))
    axs[1].set_xlabel(f"{epoch_name.capitalize()} time (s)", fontsize=fontsize)
    axs[1].set_ylabel("Gini coefficient\nof wPLI between channels")

    # Remove the third subplot axes for the legend
    axs[2].axis('off')

    # Create the legend in the third subplot
    legend_lines, legend_labels = axs[1].get_legend_handles_labels()
    axs[2].legend(legend_lines, legend_labels, loc='center', fontsize=fontsize)

    # Adjust layout
    plt.subplots_adjust(wspace=0.3, hspace=0.1, left=0.1, right=0.95, bottom=0.2, top=0.9)

# def plot_transition_graph(transition_matrix, node_dwell_time, threshold=0.2, node_scale=500, edge_scale=10, ax=None):
#     """
#     Plots a directed transition graph based on a given transition probability matrix.

#     Parameters:
#     - transition_matrix (numpy.ndarray): A square matrix representing transition probabilities between states.
#     - threshold (float): Minimum transition probability to be included in the graph (default: 0.2).
#     - node_scale (float): Scaling factor for node sizes based on self-transitions (default: 500).
#     - edge_scale (float): Scaling factor for edge widths based on transition probabilities (default: 10).
#     """

#     num_states = transition_matrix.shape[0]  # Number of states (clusters)
#     G = nx.DiGraph()  # Create a directed graph

#     # Extract self-transitions for node size scaling
#     node_sizes = 10 + (node_dwell_time * node_scale)  # Scale for visibility

#     # Add edges based on transition probabilities
#     for i in range(num_states):
#         for j in range(num_states):
#             if i != j and transition_matrix[i, j] > threshold:  # Only include significant transitions
#                 G.add_edge(i, j, weight=transition_matrix[i, j])

#     # Use a circular layout for better visualization
#     pos = nx.circular_layout(G)

#     # Extract edge weights for line thickness
#     edge_widths = [G[u][v]['weight'] * edge_scale for u, v in G.edges()]

#     # Plot the graph
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6, 6))

#     nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray",
#             node_size=node_sizes, font_size=12, arrows=True, 
#             width=edge_widths, connectionstyle="arc3,rad=0.2", ax=ax)

#     # Draw edge labels (transition probabilities)
#     edge_labels = {(i, j): f"{d['weight']:.2f}" for i, j, d in G.edges(data=True)}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def get_fixed_positions(num_states, layout='circular'):
    """
    Returns a fixed layout for a given number of states.
    
    Parameters:
        num_states (int): Number of states (nodes).
        layout (str): 'circular' to arrange nodes in a circle or 'linear' to arrange nodes in a horizontal line.
    
    Returns:
        dict: A dictionary mapping node indices to positions.
    """
    if layout == 'circular':
        angles = np.linspace(0, 2 * np.pi, num_states, endpoint=False)
        return {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}
    elif layout == 'linear':
        # Arrange nodes in a horizontal line: x evenly spaced between 0 and 1, y fixed at 0.5
        if num_states == 1:
            return {0: (0.5, 0.5)}
        return {i: (i / (num_states - 1), 0.5) for i in range(num_states)}
    else:
        raise ValueError("Unsupported layout type. Choose either 'circular' or 'linear'.")
    
def plot_transition_graph(transition_matrix, node_weight, threshold=0.2, node_labels=None, node_scale=500, edge_scale=10, print_edge_label=False, layout='circular', ax=None):
    """
    Plots a directed transition graph based on a given transition probability matrix.

    Parameters:
    - transition_matrix (numpy.ndarray): A square matrix representing transition probabilities between states.
    - node_weight (numpy.ndarray): A vector of node weights (e.g., self-transitions).
    - threshold (float): Minimum transition probability to be included in the graph (default: 0.2).
    - node_labels (dict, optional): A dictionary mapping node indices to custom labels. If None, default indices are used.
    - node_scale (float): Scaling factor for node sizes based on self-transitions (default: 500).
    - edge_scale (float): Scaling factor for edge widths based on transition probabilities (default: 10).
    """
    
    num_states = transition_matrix.shape[0]
    G = nx.DiGraph()

    # Ensure all nodes are added, even if they have no outgoing edges
    for i in range(num_states):
        G.add_node(i)

    # Add edges based on transition probabilities
    for i in range(num_states):
        for j in range(num_states):
            if i != j and transition_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=transition_matrix[i, j])

    # Use a fixed layout instead of a dynamic one
    fixed_positions = get_fixed_positions(num_states, layout)

    # Extract node sizes (ensuring the correct length)
    if isinstance(node_weight, np.ndarray) and len(node_weight) == num_states:
        node_sizes = 10 + (node_weight * node_scale)
    else:
        node_sizes = np.full(num_states, 10)  # Default size if node_weight is incorrect

    # Extract edge weights for line thickness
    edge_widths = [G[u][v]['weight'] * edge_scale for u, v in G.edges()] if G.edges else []

    if node_labels is None:
        node_labels = {i:str(i) for i in range(num_states)} # default to numeric labels

    # Plot the graph
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    nx.draw(G, fixed_positions, with_labels=True, node_color="lightblue", edge_color="gray",
            node_size=node_sizes, font_size=12, arrows=True, arrowstyle='->', 
            width=edge_widths, connectionstyle="arc3,rad=0.2", labels=node_labels, ax=ax)

    # Draw edge labels
    if print_edge_label is True:
        edge_labels = {(i, j): f"{d['weight']:.2f}" for i, j, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, fixed_positions, edge_labels=edge_labels, font_size=10, ax=ax)

def generate_region_acronyms(region_names):
    """
    Generates a dictionary of acronyms for given region names.
    
    Args:
        region_names (list of str): List of full region names (e.g., ['frontal', 'prefrontal']).
        
    Returns:
        dict: A dictionary mapping full region names to their acronyms.
    """
    if isinstance(region_names, str):
        region_names = [region_names]

    acronyms = {}
    for name in region_names:
        if name.startswith('prefrontal'):
            acronyms[name] = 'PF'
        elif name.startswith('frontal'):
            acronyms[name] = 'F'
        elif name.startswith('central'):
            acronyms[name] = 'C'
        elif name.startswith('temporal'):
            acronyms[name] = 'T'
        elif name.startswith('occipital'):
            acronyms[name] = 'O'
        elif name.startswith('parietal'):
            acronyms[name] = 'P'
        else:
            acronyms[name] = name[:2].upper()  # Fallback: Use the first two letters as a default
    return acronyms

def generate_greek_band_names(frequency_bands):
    """
    Generates a dictionary mapping frequency band names to Greek symbols or descriptive names.
    
    Args:
        frequency_bands (list of str): List of frequency band names (e.g., ['delta', 'theta', 'alpha', 'beta', 'bb']).
    
    Returns:
        dict: A dictionary mapping frequency band names to their Greek symbols or descriptive names.
    """
    if isinstance(frequency_bands, str):
        frequency_bands = [frequency_bands]

    greek_names = {
        'delta': 'δ',
        'theta': 'θ',
        'alpha': 'α',
        'low alpha': 'αl',
        'high alpha': 'αh',
        'beta': 'β',
        'low beta': 'βl',
        'high beta': 'βh',
        'gamma': 'γ'
    }
    return {band: greek_names.get(band, band.capitalize()) for band in frequency_bands}

def bottom_left_off_diagonal(array):
    """Extracts off-diagonal elements from the bottom-left rectangle of a matrix."""

    rows, cols = array.shape
    result = []

    for i in range(1, rows):
        for j in range(0, i):
            result.append(array[i, j])
    return np.array(result)