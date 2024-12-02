# %%
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import scipy.io
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from src.eeg_analysis.preprocessing.eeg_file import EEGFile
from src.eeg_analysis.preprocessing.eeg_preprocessor import EEGPreprocessor
from src.eeg_analysis.analysis.power_spectral import PowerSpectralAnalysis, TimeFrequencyRepresentation
from src.eeg_analysis.visualization.plots import plot_continuous_epochs
from src.eeg_analysis.visualization.my_custom_style import set_custom_style

# %%
dataset_dir = 'E:/Caffeine_data'
participant_names = [f'CA-{id:02}' for id in range(25, 26)]

# %%
def get_eeg_file(dataset_dir=None, participant_name=None):

    if (dataset_dir is not None) & (participant_name is not None):
        eeg_path = Path(f'{dataset_dir}/{participant_name}/{participant_name}').with_suffix('.eeg')
    else: 
        # Create a Tkinter root window (it will not be shown)
        root = tk.Tk()
        root.withdraw()
        
        # Set the file dialog to appear in front
        root.attributes('-topmost', True)

        # Ask the user to select a .eeg file
        eeg_file_path = filedialog.askopenfilename(title="Select EEG File", filetypes=[("EEG files", "*.eeg")])
        if not eeg_file_path:
            raise FileNotFoundError("No file selected.")
        
        # Convert to Path object for easier manipulation
        eeg_path = Path(eeg_file_path)
    
    base_path = eeg_path.with_suffix('')
    base_dir = eeg_path.parent
    
    # Extract participant ID (assuming filename is `CA-XX.eeg`)
    participant_name = base_path.name
    participant_id = int(participant_name.split('-')[1])  # Adjust this if your file naming convention is different
    
    # Generate paths for the corresponding files
    vhdr_path = base_path.with_suffix('.vhdr')
    vmrk_path = base_path.with_suffix('.vmrk')

    return participant_id, base_dir, vhdr_path, vmrk_path, eeg_path


for participant_name in participant_names:
    print(participant_name)
        
    try:
        participant_id, base_dir, vhdr_path, vmrk_path, eeg_path = get_eeg_file(dataset_dir, participant_name)

        # Make a directory for analysis results
        base_results_dir = base_dir / 'analysis_results'
        if not base_results_dir.exists():
            base_results_dir.mkdir(parents=True, exist_ok=True)

        eeg_file = EEGFile(participant_id, str(vhdr_path), str(vmrk_path), str(eeg_path))
        eeg_file.read_vhdr()

        # %% [markdown]
        # ##### Load preprocessed .mat EEG data

        # %%
        def load_mat_file(participant_id, epoch, variables, all_channel_names):

            base_dir = Path(r'D:/Anesthesia_Research_Fellow/preprocessed_EEG_by_DuanLI')
            base_filename = f'CA_{participant_id:02}_{epoch}_denoised.mat'
            file_path = base_dir / base_filename

            # Check if the file exists
            if not file_path.exists():
                raise FileNotFoundError(f"The file {file_path} does not exist")

            # Load .mat file
            mat_data = scipy.io.loadmat(file_path)

            # Access the variables
            data={}
            for variable in variables:
                if variable in mat_data:
                    curr_variable = mat_data[variable]
                    if variable in ['zz_epoch', 'zz_noise']:
                        data[variable] = np.transpose(curr_variable, [1, 0])
                    elif variable in ['channels_remained', 'Index_tt']:
                        data[variable] = curr_variable.flatten() - 1 # Due to Python zero-indexing
                    elif variable in ['fs', 'L_raw', 'L_denoised']:
                        data[variable] = curr_variable.flatten()[0]
                    else:
                        data[variable] = curr_variable.flatten()
                else:
                    print(f"The variable '{variable}' does not exist in the .mat file for epoch '{epoch}'")
            
            # channels_remained = data['channels_remained'].flatten().tolist()
            # data['channel_names'] = [all_channel_names[i] for i in channels_remained]
            # data['omitted_channel_names'] = [item for item in all_channel_names if (item not in data['channel_names']) & (len(item) <= 3)]

            return data

        # Example for loading different epochs for participant_id = 1
        epoch_names = ['preOP_epoch', 'preExtube', 'PACU_epoch']
        new_epoch_names = {
            'preOP_epoch': 'preop_rest',
            'preExtube': 'emergence',
            'PACU_epoch': 'pacu_rest'
        }
        variables = ['channels_remained', 'fs', 'L_denoised', 'L_raw', 'tt', 'Index_tt', 'zz_epoch', 'zz_noise']
        all_channel_names = eeg_file.channel_names

        epoch_data = {}
        for epoch_name in epoch_names:
            try:
                epoch_data[new_epoch_names[epoch_name]] = load_mat_file(participant_id, epoch_name, variables, all_channel_names)
                
                # Process the loaded data as needed
            except FileNotFoundError as e:
                print(e)


        # %% [markdown]
        # #### Zero padding the eeg data, so the time dimension is linear (with no missing point)

        # %%
        total_num_eeg_channels = 16

        epoch_names = list(epoch_data.keys())
        for epoch_name in epoch_names:
            if epoch_name == 'emergence':
                curr_epoch_data = epoch_data[epoch_name]
                tt = curr_epoch_data['tt']
                min_tt = np.min(tt)
                max_tt = np.max(tt)
                
                complete_time_range = np.arange(min_tt, max_tt + 1)
                time_indices = tt - min_tt  # These are the positions where `tt` values should go

                # Preallocate zero-padded EEG data arrays
                for eeg_data_name in ['zz_epoch', 'zz_noise']:
                    curr_eeg_data = curr_epoch_data[eeg_data_name]
                    zero_padded_eeg = np.zeros((len(complete_time_range), curr_eeg_data.shape[1]))

                    # Utilize NumPy indexing to fill in the non-zero data
                    zero_padded_eeg[time_indices] = curr_eeg_data

                    epoch_data[epoch_name][eeg_data_name] = zero_padded_eeg

                epoch_data[epoch_name]['tt'] = complete_time_range

            else:
                epoch_data[epoch_name]['tt'] = epoch_data[epoch_name]['Index_tt']
                del epoch_data[epoch_name]['Index_tt']


            curr_epoch_data = epoch_data[epoch_name]
            curr_channels_remained = curr_epoch_data['channels_remained']
            for eeg_data_name in ['zz_epoch']: #, 'zz_noise'
                zero_padded_channels = np.zeros((curr_epoch_data['zz_epoch'].shape[0], total_num_eeg_channels))

                for i_chan, chan in enumerate(curr_channels_remained):
                    zero_padded_channels[:, chan] = curr_epoch_data[eeg_data_name][:, i_chan]

                epoch_data[epoch_name][eeg_data_name] = zero_padded_channels

            epoch_data[epoch_name]['eliminated_channels'] = [i for i in range(total_num_eeg_channels) if i not in curr_channels_remained.tolist()]
            del epoch_data[epoch_name]['channels_remained']

        # %%
        plt.figure()
        plt.plot(epoch_data['emergence']['zz_epoch'][:500, 6]) # epoch_data['emergence']['zz_epoch']

        # %% [markdown]
        # #### Generate .eeg file from the imported data for visualizing the eeg using Neuroscope

        # %%
        epoch_name = 'emergence'
        eeg_data = epoch_data[epoch_name]['zz_epoch']

        sampling_rate = epoch_data[epoch_name]['fs']

        sufffix = f'_{epoch_name}'
        eeg_int = EEGPreprocessor.convert_to_neuroscope_eeg(eeg_data, sampling_rate, eeg_path, sufffix)

        # %% [markdown]
        # ##### Modify the eeg_file object using the preprocessed data

        # %%
        epoch_names = epoch_data.keys()

        eeg_data = {}
        for epoch_name in epoch_names:
            eeg_data[epoch_name] = epoch_data[epoch_name]['zz_epoch']

        eeg_file.eeg_data = eeg_data
        eeg_file.sampling_frequency = epoch_data[epoch_name]['fs'] 

        # %% [markdown]
        # ### Note that the 'emergence' epoch now defined as 10 minutes prior to drug infusion until the extubation 

        # %%
        eeg_file.eeg_data.keys()

        # %%
        power_spectral = PowerSpectralAnalysis(eeg_file, window_size=2, step_size=1) 

        # %%
        power_spectral.calculate_time_frequency_map(select_channels = None, select_epochs=['emergence'], method='multitaper')

        # %%
        
        set_custom_style()

        epochs_to_include = ['emergence']
        # Create a new dictionary with only the specified epochs
        tfr_to_plot = {key: value for key, value in power_spectral.tfr.items() if key in epochs_to_include}

        start_time = 0
        end_time = 0
        for _, tfr in tfr_to_plot.items():
            end_time = end_time + tfr.times[-1]

        # Plotting for all 16 channels
        channels_to_plot = list(range(16))
        chan_names = power_spectral.select_channel_names
        num_channels = len(channels_to_plot)

        group_names = list(power_spectral.channel_groups.keys())
        num_groups = len(group_names)

        # Define single-column or double_column width figure
        fig_width = 600 # Adjusted to fit the 6x3 subplot grid
        fig_height = 300 # Adjusted to fit the 6x3 subplot grid

        font_size = 6

        fig = plt.figure()
        fig.set_size_inches([fig_width/72, fig_height/72])
        gs = GridSpec(
            nrows=6, 
            ncols=4, 
            figure=fig, 
            wspace = 0.4, 
            hspace = 0.4,
            width_ratios=[1,1,1,0.05]
        )

        for i_region, region_name in enumerate(group_names):
            region_channels_indices = [chan_names.index(name) for name in chan_names if name in power_spectral.channel_groups[region_name]]
            for i_channel, channel_idx in enumerate(region_channels_indices):
                row = i_region
                col = i_channel
                ax = fig.add_subplot(gs[row, col])

                im = plot_continuous_epochs(tfr_to_plot, ax = ax, colorbar=False, vmin=0, vmax=50, channel = channel_idx, cmap = 'inferno') # vmin=0, vmax = 50 

                if row == 5:  # Label x-axis on the bottom row
                    ax.set_xlabel('Time (s)')
                else:
                    ax.set_xticklabels([])
                
                ax.set_xticks(np.arange(0, end_time, 1000))
                ax.set_yticks(np.arange(0, 55, 10))
                ax.set_ylabel('Frequency (Hz)', labelpad=0.5)
                ax.set_title(chan_names[channel_idx], pad = 0.5, loc='left')
                ax.tick_params(axis='both', which='major', labelsize=5, length=2, width=0.5, pad=1)

        # Add colorbar
        cax = fig.add_subplot(gs[0, -1])
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label('Power (dB)', fontsize=5)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=5)

        plt.subplots_adjust(left=0.1, right= 0.9, bottom=0.1, top=0.9)

        # Save plot
        base_results_filename = base_results_dir / f'CA-{participant_id:02}'
        filename = f'{base_results_filename}_tfrs_preprocessed.pdf'
        file_path = os.path.join(base_results_dir, filename)
        # plt.savefig(file_path, format='pdf', dpi=300)
    
        # plt.show()
        plt.close()

        # %%
        power_spectral.postprocess_anatomical_region_average()

        # %%
        window_size = 60
        step_size = 30

        power_spectral.postprocess_time_window_average(window_size=window_size, step_size=step_size, attr_name='region_average_tfr')

        # %% [markdown]
        # ##### Calculate FOOOF parameters

        # %%
        power_spectral.postprocess_aperiodic_paramaters(attr_name='window_average_tfr', freq_range=[30, 40])

        # %%
        power_spectral.postprocess_periodic_parameters(attr_name='window_average_tfr', peak_threshold= 1, overlap_threshold=0.5)

        # %%
    
        set_custom_style()

        regions = ['prefrontal', 'frontal', 'central', 'temporal', 'parietal', 'occipital']
        epoch_name = 'emergence'

        frequency_bands = ['delta', 'alpha', 'beta', 'gamma']
        greek_band_names = {
            'delta': r'$\delta$',
            'alpha': r'$\alpha$',
            'beta': r'$\beta$',
            'gamma': r'$\gamma$'
        }

        # Number of regions
        num_regions = len(regions)

        # Determine the number of windows from the first region (assuming all regions have the same shape)
        num_windows = len(power_spectral.fooof_models[regions[0]][epoch_name])

        # Determine the grid size for the subplots
        cols_windows = int(np.ceil(np.sqrt(num_windows)))
        rows_windows = int(np.ceil(num_windows / cols_windows))

        # Create a main figure with subplots for each region
        fig_width = 6
        fig_height = 3.5 * num_regions + 0.5 * num_regions
        fig = plt.figure()
        fig.set_size_inches([fig_width, fig_height])
        gs = GridSpec(
            nrows=num_regions * (rows_windows + 2),  # One additional row per region for the title
            ncols=cols_windows,
            figure=fig,
            wspace=0.2,
            hspace=0.3
        )

        # Utility to set common y-limits across all plots for a region
        # def get_common_y_limits(region, num_windows, epoch_name):
        #     all_ylimits = []
        #     for win_idx in range(num_windows):
        #         model = power_spectral.fooof_models[region][epoch_name][0, win_idx]
        #         ylimits = model.plot(plt_log=False, plot_peaks=False, add_legned=False, linewidth=0.5, return_limits=True)
        #         all_ylimits.append(ylimits)
        #     common_ylim = np.min(all_ylimits), np.max(all_ylimits)
        #     return common_ylim

        # Plot each window for each region
        for r_idx, region in enumerate(regions):
            # Add a title for each region spanning all columns
            ax_title = fig.add_subplot(gs[r_idx * (rows_windows + 2)+1, :])
            ax_title.set_title(region.capitalize(), fontsize=10, fontweight='bold', pad=10, loc='left')
            ax_title.axis('off')
            
            # Get common y-limits for the current region
            # common_ylim = get_common_y_limits(region, num_windows, epoch_name)
            curr_fooof_models = power_spectral.fooof_models[region][epoch_name]
            if curr_fooof_models != {}:
                for win_idx in range(num_windows):
                    ax = fig.add_subplot(gs[r_idx * (rows_windows + 2) + (win_idx // cols_windows) + 1, win_idx % cols_windows])
                    curr_fooof_models[0, win_idx].plot(plt_log=False, plot_peaks=False, add_legned=False, linewidth=0.3, ax=ax)
                    
                    # Set common y-limits
                    # ax.set_ylim(common_ylim)
                    ax.set_ylim([0, 5.5])
                    ax.set_xticks(np.arange(0, 51, 10))

                    yl = ax.get_ylim()
                    for idx, fband in enumerate(frequency_bands): 
                        curr_cf = power_spectral.fooof_periodic_parameters[region][epoch_name][fband]['cf'][0][win_idx]
                        ax.axvline(x=curr_cf, color='green', linestyle='-', linewidth=0.3, alpha=0.5)
                        ax.annotate(f'{greek_band_names[fband]} {curr_cf:.1f} Hz', xy=(curr_cf-8, yl[0]+0.05*np.diff(yl)), xytext=(5, 0),
                                textcoords='offset points', rotation=90, va='bottom', ha='left', fontsize=3, color='green', alpha=0.5)

                    ax.grid(which='both', linewidth=0.25)
                    
                    # Set labels based on position
                    if win_idx % cols_windows == 0:  # Leftmost column
                        ax.set_ylabel(ax.get_ylabel(), fontsize=5)
                        yticks = ax.get_yticks()
                        ax.set_yticks(yticks)
                        ax.set_yticklabels([f'{y:.1f}' for y in yticks], fontsize=5)
                    else:
                        ax.set_ylabel('')
                        ax.set_yticklabels([])

                    if win_idx // cols_windows == rows_windows - 1 or ((win_idx // cols_windows == rows_windows - 2) and (win_idx >= num_windows - cols_windows)):  # Last row or second last row with empty bottom
                        ax.set_xlabel(ax.get_xlabel(), fontsize=5)
                        xticks = ax.get_xticks()
                        ax.set_xticks(xticks)
                        ax.set_xticklabels([f'{x:.1f}' for x in xticks], rotation=45, fontsize=5)
                    else:
                        ax.set_xlabel('')
                        ax.set_xticklabels([])

                    # Remove the legend if it exists
                    legend = ax.get_legend()
                    if legend:
                        legend.remove()

                    ax.set_ylim([0, 5.5])

                    ax.set_title(f'win {win_idx}', fontsize=5, fontweight='regular', loc='center', pad=0)

        # Adjust layout
        plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.9)

        # Save plot
        base_results_filename = base_results_dir / f'CA-{participant_id:02}'
        filename = f'{base_results_filename}_fooof_models_time_resolved_preprocessed.pdf'
        file_path = os.path.join(base_results_dir, filename)
        # plt.savefig(file_path, format='pdf', dpi=300)
    
        # plt.show()
        plt.close()

        # power_spectral.fooof_models['prefrontal']['emergence'][0, 5].plot(plt_log=False, plot_peaks=False)

        # %%
        participant_id

        # %%
        regions = ['prefrontal', 'frontal', 'central', 'temporal', 'parietal', 'occipital']
        epochs = ['emergence']
        frequency_bands = ['delta', 'alpha', 'beta', 'gamma']
        periodic_features = ['cf', 'amplitude', 'bw', 'avg_power_absolute', 'avg_power_relative']
        aperiodic_features = ['exponents', 'intercepts']
        num_regions = len(regions)
        num_bands = len(frequency_bands)
        num_periodic_features = len(periodic_features)
        num_aperiodic_features = len(aperiodic_features)

        epoch_colors = ['#007377', '#483D8B', '#C84E00', '#556B2F']

        greek_band_names = {
            'delta': r'$\delta$',
            'alpha': r'$\alpha$',
            'beta': r'$\beta$',
            'gamma': r'$\gamma$'
        }

        full_feature_names = {
            'cf': 'central\nfrequency',
            'amplitude': 'amplitude',
            'bw': 'bandwidth',
            'avg_power_relative': 'mean\nrel. power',
            'avg_power_absolute': 'mean\nabs. power'
        }

        # Calculate an approximate figure height
        fig_width = 6  # inches
        fig_height = 3 * num_regions + 0.5 * num_regions  # inches (additional space for region titles)

        font_size = 6  # font size for the titles

        # Create a figure
        fig = plt.figure()
        fig.set_size_inches([fig_width, fig_height])
        gs = GridSpec(
            nrows=num_regions * (num_bands + 2),  # accounting for extra title rows
            ncols=num_periodic_features+1,
            figure=fig,
            wspace=0.5, 
            hspace=0.2
        )

        # Loop through each region, band, and feature to create plots
        for r_idx, region in enumerate(regions):
            # Create an empty axis for the supertitle of the region
            ax_title = fig.add_subplot(gs[r_idx * (num_bands + 2)+1, :])  # Span all columns
            ax_title.set_title(region.capitalize(), fontsize=10, fontweight='bold', loc='left', pad=18)
            ax_title.axis('off')  # Hide the axis for the region title
            
            for b_idx, band in enumerate(frequency_bands):
                for f_idx, feature in enumerate(periodic_features):
                    # Adjust the row index by adding 1 to account for the region title row
                    ax = fig.add_subplot(gs[r_idx * (num_bands + 2) + b_idx + 1, f_idx])
                    
                    # Example data retrieval and plotting (replace accordingly)
                    start_shift = 0
                    for epoch_idx, epoch in enumerate(epochs):
                        parameter_data = power_spectral.fooof_periodic_parameters[region][epoch][band][feature][0]
                        window_centers = start_shift + power_spectral.window_average_tfr[region][epoch].times
                        start_shift += power_spectral.tfr[epoch].times[-1]

                        # Plot the data on the subplot
                        if not np.all(np.isnan(parameter_data)):
                            ax.plot(window_centers, parameter_data, linewidth=0.75, color='k', marker='none', markersize=.5) #epoch_colors[epoch_idx]
                            ax.plot(window_centers, [np.nanmedian(parameter_data)]*len(window_centers), linewidth=0.5, linestyle=':', color='k')
                            
                            # Set a dashed vertical line to indicate a change of period
                            # ax.axvline(x=start_shift, color='grey', linestyle='--', linewidth=0.5)

                    ax.tick_params(axis='both', which='major', labelsize=5, length=2, width=0.5, pad=1)
                    ax.set_xticks(range(0, int(window_centers[-1]), 1000))
                    ax.set_xticklabels(ax.get_xticks(), rotation=45)

                    # Label the subplot if it's the first column or the last row within a region
                    if f_idx == 0:
                        ax.set_ylabel(f'{greek_band_names[band]}', labelpad=5, color='k', fontweight='bold', fontsize=6)
                    if b_idx == num_bands - 1:
                        ax.set_xlabel('Time (s)', labelpad=0)
                    if b_idx != num_bands - 1:
                        ax.set_xticklabels([])
                    if b_idx == 0:
                        ax.set_title(f'{full_feature_names[feature].capitalize()}', pad=0)
                    

            # Now let's add the exponents and intercepts plots at the end of each region's row span
            # Plot exponents in the next-to-last column
            for ap_idx, aperodic_param in enumerate(aperiodic_features):
                ax = fig.add_subplot(gs[r_idx * (num_bands + 2) + ap_idx+1, -1])

                start_shift = 0
                for epoch_idx, epoch in enumerate(epochs):
                    parameter_data = power_spectral.fooof_aperiodic_parameters[region][epoch][aperodic_param][0]
                    window_centers = start_shift + power_spectral.window_average_tfr[region][epoch].times
                    start_shift += power_spectral.tfr[epoch].times[-1]

                    if np.any(~np.isnan(parameter_data)):
                        ax.plot(window_centers, parameter_data, linewidth=0.75, color='k', label=epoch, marker='none', markersize=.5)
                        ax.plot(window_centers, [np.nanmedian(parameter_data)]*len(window_centers), linewidth=0.5, linestyle=':', color='k')

                    # ax.axvline(x=start_shift, color='grey', linestyle='--', linewidth=0.5)
            
                ax.set_ylabel(aperodic_param[:-1].capitalize(), labelpad=0)
                ax.tick_params(axis='both', which='major', labelsize=5, length=2, width=0.5, pad=1)
                ax.set_xticks(range(0, int(window_centers[-1]), 1000))
                ax.set_xticklabels(ax.get_xticks(), rotation=45)

                ylim = ax.get_ylim()
                emergence_data = power_spectral.fooof_aperiodic_parameters[region]['emergence'][aperodic_param][0]
                if np.any(~np.isnan(emergence_data.flatten())):
                    ax.set_ylim([np.nanpercentile(emergence_data, 10), ylim[1]])

                if ap_idx == num_aperiodic_features - 1:
                    ax.set_xlabel('Time (s)', labelpad=0)
                if ap_idx != num_aperiodic_features - 1:
                    ax.set_xticklabels([])

                if (r_idx == 0) and (ap_idx == 0):
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

        # Adjust layout
        plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.9)

        # Save plot
        base_results_filename = base_results_dir / f'CA-{participant_id:02}'
        filename = f'{base_results_filename}_fooof_results_time_resolved_preprocessed_slope_30-40Hz.pdf'
        file_path = os.path.join(base_results_dir, filename)
        plt.savefig(file_path, format='pdf', dpi=300)

        # plt.show()
        plt.close()

        # %%
        power_spectral.postprocess_frequency_domain_whitening(attr_name='region_average_tfr')

        # %%
        # For the prefrontal area plot the region average time-frequency power together with the power in delta and alpha frequency band over time 

        region_average_tfr = power_spectral.whitened_tfr
        num_regions = len(region_average_tfr)

        epoch_name = 'emergence'
        region_name = 'prefrontal'

        set_custom_style()
        # Define single-column or double_column width figure
        fig_width = 200 # Adjusted to fit the 6x3 subplot grid
        fig_height = 80 # Adjusted to fit the 6x3 subplot grid

        # Define a color map for the frequency bands
        frequency_bands = list(power_spectral.fooof_periodic_parameters['prefrontal']['emergence'].keys())

        # Create a sequential colormap, 'viridis' for example, excluding the broadband
        sequential_cmap = plt.cm.YlOrRd(np.linspace(0.35, 1, len(frequency_bands) - 1))
        color_map = {band: sequential_cmap[i] for i, band in enumerate(frequency_bands[:-1])}

        font_size = 6

        fig = plt.figure()
        fig.set_size_inches([fig_width/72, fig_height/72])
        gs = GridSpec(
            nrows=2, 
            ncols=2, 
            figure=fig,  
            hspace = 0.2,
            width_ratios=[1,0.05]
        )

        # Plot region-average TFR for the given region
        ax = fig.add_subplot(gs[0, 0])
        im = plot_continuous_epochs(region_average_tfr['prefrontal'], ax=ax, vmin=-5, vmax=10, colorbar=False, cmap='inferno', title=region_name)     
        ax.set_ylabel('Frequency\n(Hz)')
        ax.set_title(region_name.capitalize(), pad = 0.5, loc='left')

        ax.set_xticks(range(0, int(region_average_tfr['prefrontal']['emergence'].times[-1]), 1000))
        ax.set_xticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=5, length=2, width=0.5, pad=1)

        ax.set_ylim([0, 40])
        ax.set_yticks(range(0, 41, 10))

        # Add colorbar
        cax = fig.add_subplot(gs[0, -1])
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label('Power (dB)', fontsize=5)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=5)

        # Plot overlaid power of delta and alpha frequency bands over time
        window_centers = power_spectral.window_average_tfr[region][epoch].times
        ax_time_series = fig.add_subplot(gs[1, 0])

        # all_normal_data = []
        for fband in ['delta', 'alpha', 'beta']:
            feature='avg_power_absolute' if fband == 'delta' else 'amplitude'
            parameter_data = power_spectral.fooof_periodic_parameters[region_name][epoch_name][fband][feature][0]
            parameter_data = (parameter_data - np.nanmean(parameter_data))/np.nanstd(parameter_data)

            # pre_drug_infusion_avg = np.nanmean(parameter_data[:6])

            # Plot the data on the subplot
            if (not np.all(np.isnan(parameter_data))):# & (~np.isnan(pre_drug_infusion_avg)):
                # normalized_parameter_data = parameter_data/pre_drug_infusion_avg
                # all_normal_data.append(normalized_parameter_data)
                ax_time_series.plot(window_centers, parameter_data, linewidth=0.5, color=color_map[fband], marker='none', label=greek_band_names[fband], markersize=.5) #epoch_colors[epoch_idx]

        ax_time_series.set_xlim(ax.get_xlim())
        ax_time_series.set_xticks(range(0, int(region_average_tfr['prefrontal']['emergence'].times[-1]), 1000))

        #set_the_ylim
        # all_normal_data = np.concatenate(all_normal_data, axis=0)
        # all_normal_data = all_normal_data[~np.isnan(all_normal_data)]
        # all_normal_data = (all_normal_data - np.mean(all_normal_data))/np.std(all_normal_data)
        # yl = [0, 1.05*np.percentile(all_normal_data, 95)]

        # ax_time_series.set_ylim(yl)
        yl = ax_time_series.get_ylim()

        ax_time_series.set_yticks(np.arange(yl[0], yl[1], 1))
        ax_time_series.set_ylabel('Power\n(normalized)')
        ax_time_series.set_xlabel ('Time (s)')

        # Add the rectangle patch to indicate the pre-drug infusion region
        start_time = window_centers[0] - (step_size/2)
        end_time = window_centers[5] + (step_size/2)
        rect = Rectangle((start_time, yl[0]), end_time - start_time, yl[1]-yl[0], facecolor='gray', alpha=0.2)
        ax_time_series.add_patch(rect)

        ax.tick_params(axis='both', which='major', labelsize=5, length=2, width=0.5, pad=1)


        # Create the legend 
        legend_ax = fig.add_subplot(gs[1, -1])
        legend_ax.axis('off')
        legend_lines, legend_labels = ax_time_series.get_legend_handles_labels()
        legend_ax.legend(legend_lines, legend_labels, loc='center', fontsize=font_size)


        plt.subplots_adjust(left=0.2, right= 0.8, bottom=0.2, top=0.9)

        # Save plot
        base_results_filename = base_results_dir / f'CA-{participant_id:02}'
        filename = f'{base_results_filename}_prefrontal_emergence_trajectory.pdf'
        file_path = os.path.join(base_results_dir, filename)
        # plt.savefig(file_path, format='pdf', dpi=300)

        # plt.show()
        plt.close()

        # %%
        region_average_tfr = power_spectral.whitened_tfr
        num_regions = len(region_average_tfr)

        set_custom_style()
        epochs_to_include = ['emergence']
        # Create a new dictionary with only the specified epochs
        tfr_to_plot = {key: value for key, value in power_spectral.tfr.items() if key in epochs_to_include}
        select_epochs = ['emergence']

        # freq_bands = power_spectral.fooof_periodic_parameters['prefrontal'][select_epochs[0]].keys()

        # Define single-column or double_column width figure
        fig_width = 200 # Adjusted to fit the 6x3 subplot grid
        fig_height = 300 # Adjusted to fit the 6x3 subplot grid

        font_size = 6

        fig = plt.figure()
        fig.set_size_inches([fig_width/72, fig_height/72])
        gs = GridSpec(
            nrows=6, 
            ncols=2, 
            figure=fig, 
            wspace = 0.4, 
            hspace = 0.4,
            width_ratios=[1,0.05]
        )

        region_count = 0
        for region, tfr in region_average_tfr.items(): # region_average_tfr
            ax = fig.add_subplot(gs[region_count, 0])
            im = plot_continuous_epochs(tfr, ax=ax, colorbar=False, vmin=-5, vmax=10, cmap='inferno', title=region) 

            if region == list(region_average_tfr.keys())[-1]:
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticklabels([])
                
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(region.capitalize(), pad = 0.5, loc='left')

            
            # for band in freq_bands:
            start_shift = 0
            for epoch in select_epochs:
                window_centers = start_shift + power_spectral.window_average_tfr[region][epoch].times

                # freq_band_info = power_spectral.fooof_periodic_parameters[region][epoch][band]
                # cf = freq_band_info['cf'][0]
                # lower_bound = freq_band_info['lower_bound'][0]
                # upper_bound = freq_band_info['upper_bound'][0]
                
                # # ax.plot(window_centers, cf, linestyle='-', linewidth=0.2, color='w')
                # # ax.plot(window_centers, lower_bound, linestyle=':', linewidth=0.2, color='w')  # Dotted line style
                # # ax.plot(window_centers, upper_bound, linestyle=':', linewidth=0.2, color='w')  # Dotted line style

                start_shift += power_spectral.tfr[epoch].times[-1]

            ax.set_xticks(range(0, int(window_centers[-1]), 1000))
            ax.tick_params(axis='both', which='major', labelsize=5, length=2, width=0.5, pad=1)

            ax.set_ylim([0, 55])
            ax.set_yticks(range(0, 55, 10))
            region_count += 1

        # Add colorbar
        cax = fig.add_subplot(gs[0, -1])
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label('Power (dB)', fontsize=5)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=5)

        plt.subplots_adjust(left=0.2, right= 0.8, bottom=0.1, top=0.9)

        # Save plot
        base_results_filename = base_results_dir / f'CA-{participant_id:02}'
        filename = f'{base_results_filename}_tfrs_region_averages_preprocessed.pdf'
        file_path = os.path.join(base_results_dir, filename)
        # plt.savefig(file_path, format='pdf', dpi=300)

        # plt.show()
        plt.close()
    except: 
        Warning(f'No data directory exists for participant {participant_name} or the calculation failed due to an error')