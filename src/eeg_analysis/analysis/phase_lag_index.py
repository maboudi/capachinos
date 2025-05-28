import numpy as np
from src.eeg_analysis.preprocessing.eeg_preprocessor import EEGPreprocessor
from src.eeg_analysis.utils.helpers import select_channels_and_adjust_data
from mne_connectivity import spectral_connectivity_epochs
import warnings

class PhaseLagIndex:
    def __init__(self, eeg_file):
        self.participant_id = eeg_file.participant_id
        self.eeg_data = eeg_file.eeg_data
        self.channel_names = eeg_file.channel_names
        self.channel_groups = eeg_file.channel_groups
        self.sampling_frequency = eeg_file.sampling_frequency

        # self.select_channels = None
        self.select_channel_names = None
        self.region_names = None

        self.pli_method = None
        self.window_size = None
        self.overlap_ratio = None
        self.rep_seg_size = None
        self.fmin = None
        self.fmax = None
        self.fmin_bb_avg = None
        self.fmax_bb_avg = None
        self.mt_bandwidth = None

        self.conn_wpli = None
        self.conn_wpli_favg = None
        self.conn_wpli_win = None
        self.conn_wpli_win_favg = None
        self.freqs = None
        self.win_centers = None

        self.conn_wpli_anat = None, 
        self.conn_wpli_favg_anat = None, 
        self.conn_wpli_win_anat = None, 
        self.conn_wpli_win_favg_anat = None
        self.frequency_bands_of_interest=None,


    def calculate_pli(
            self, 
            select_epochs = None, 
            window_size = 300, 
            overlap_ratio = 0.5, 
            rep_seg_size = 10, 
            rep_seg_overlap_ratio = 0,
            mt_bandwidth = 2,
            pli_method='wpli2_debiased',
            fmin = 0.5,
            fmax = 55,
            fmin_bb_avg = 0.5,
            fmax_bb_avg = 20, 
            frequency_bands_of_interest=None,
            select_channels = None
        ):

        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.rep_seg_size = rep_seg_size
        self.rep_seg_overlap_ratio = rep_seg_overlap_ratio
        self.mt_bandwidth = mt_bandwidth
        self.pli_method = pli_method
        self.fmin = fmin
        self.fmax = fmax
        self.fmin_bb_avg = fmin_bb_avg
        self.fmax_bb_avg = fmax_bb_avg
        self.frequency_bands_of_interest = frequency_bands_of_interest

        # If data_source is a dictionary 
        if isinstance(self.eeg_data, dict):
            epochs_to_process = self.eeg_data.keys() if select_epochs is None else select_epochs
            
            # Check if selected_epochs exist in data_source
            if not all(epoch in self.eeg_data for epoch in epochs_to_process):
                missing_epochs = [epoch for epoch in epochs_to_process if epoch not in self.eeg_data]
                raise ValueError(f'The following epochs are not available: {missing_epochs}')

            # calculate PLI for each epoch
            self.conn_wpli = {}
            self.conn_wpli_favg = {}
            self.conn_wpli_win = {}
            self.conn_wpli_win_favg = {}
            self.win_centers = {}
            for epoch_idx, epoch_name in enumerate(epochs_to_process):
                print(f'Calculating phase lag index for the {epoch_name} epoch ...')
                
                epoch_data = self.eeg_data[epoch_name]
                pli, pli_favg, pli_win, pli_win_favg, win_centers, chan_names = self._calculate_epoch_pli(epoch_data, select_channels)
                
                self.conn_wpli[epoch_name] = pli
                self.conn_wpli_favg[epoch_name] = pli_favg
                self.conn_wpli_win[epoch_name] = pli_win
                self.conn_wpli_win_favg[epoch_name] = pli_win_favg
                self.win_centers[epoch_name] = win_centers

                ## Store the results for frequency bands of interest

                if epoch_idx == 0:
                    self.select_channel_names = chan_names
                print('\nDONE!')

        # If data_source in continuous EEG data s
        else:
            pli, pli_favg, pli_win, pli_win_favg, win_centers, chan_names = self._calculate_epoch_pli(self.eeg_data, select_channels)
            
            self.conn_wpli = pli
            self.conn_wpli_favg = pli_favg
            self.conn_wpli_win = pli_win
            self.conn_wpli_win_favg = pli_win_favg
            self.win_centers = win_centers
            self.select_channel_names = chan_names

    def _calculate_epoch_pli(self, epoch_data, select_channels):

        chan_names, _, epoch_data = self._select_channels_and_adjust_data(epoch_data, select_channels)

        # Calculate pli for each epoch's eeg as a whole (no windowing)
        print('Entire duration ...')
        conn_wpli, conn_wpli_favg = self._calculate_window_pli(epoch_data, rm_outlier_segs=False)


        # Calculate pli in windows
        print(f'{self.rep_seg_size}-second windows ...')
        eeg_windows, window_centers = EEGPreprocessor.get_segments(
            epoch_data,
            self.sampling_frequency, 
            self.window_size, 
            self.overlap_ratio)
        conn_wpli_win, conn_wpli_win_favg = self._calculate_window_pli(eeg_windows, rm_outlier_segs=True)


        return conn_wpli, conn_wpli_favg, conn_wpli_win, conn_wpli_win_favg, window_centers, chan_names
    
    def _select_channels_and_adjust_data(self, epoch, select_channels):
        return select_channels_and_adjust_data(
            epoch,
            select_channels,
            self.channel_names,
            self.channel_groups
        )
    
    def _calculate_window_pli(self, eeg_windows, rm_outlier_segs=False):

        """
        Calculate phase lag index for each window or for each epoch as a whole if there is only one window

        Paramters:
        eeg_windows: EEG data formated as samples X channels X windows 
        rm_outlier_segs: Boolian parameter specifying whether to remove (zero-pad) eeg from channels displaying noisy amplitude. Perfmored separately within each segment 
        """
        
        freq_bands_oi = self.frequency_bands_of_interest
        freq_bands_oi['bb'] = [self.fmin_bb_avg, self.fmax_bb_avg]


        if eeg_windows.ndim == 2: # if eeg data is not windowed meaning entire epoch
            eeg_windows = eeg_windows[:, :, np.newaxis]

        def is_outlier(channel_data, threshold=3):
            """
            For each segments, check if there are nosiy samples by comparing the ampitudes with a thershold 
            
            Parameters:
            channel_data (ndarray): The channel data with shape (n_times,).
            threshold (float): The z-score threshold to determine outliers. Default is 3.
            
            Returns:
            bool: True if the channel data is an outlier, False otherwise.
            """

            if np.all(channel_data == 0):
                return False  # If the channel data is all zeros, it's not an outlier

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", RuntimeWarning)  # Catch all runtime warnings
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                if std == 0:
                    return True  # Treat this as an outlier since std-dev is zero (constant signal)
                z_scores = np.abs((channel_data - mean) / std)
                
                if len(w) > 0:
                    # Print the warning
                    for warning in w:
                        print(warning.message)
                        print(f'Std deviation of channel data: {std}')

            return np.any(z_scores > threshold)

        # Calculate the shape of one connectivity matrix to know dimensions for zero-padding
        sample_window, _ = EEGPreprocessor.get_segments(eeg_windows[:, :, 0], self.sampling_frequency, self.rep_seg_size, self.rep_seg_overlap_ratio)
        sample_window = np.transpose(sample_window, (2, 1, 0))
        sample_conn = spectral_connectivity_epochs(
            sample_window,
            method=self.pli_method,
            mode='multitaper',
            mt_bandwidth=self.mt_bandwidth,
            sfreq=self.sampling_frequency,
            fmin=self.fmin,
            fmax=self.fmax,
            faverage=False, # specifies whether to average over the frequecny samples
            mt_adaptive=False,
            verbose=False
        )
        conn_shape = sample_conn.get_data(output='dense').shape

        conn_wpli_win = []

        band_f_indices = {}
        conn_wpli_win_favg = {}
        for band_name in freq_bands_oi.keys():
            conn_wpli_win_favg[band_name] = []

        num_windows = eeg_windows.shape[2]

        for win_idx in range(num_windows):

            if num_windows > 1:
                print(f'window {win_idx + 1}', end='\r')
            
            window = eeg_windows[:, :, win_idx]

            # Dividing into smaller segments within each window for averaging 
            segments, _ = EEGPreprocessor.get_segments(window, self.sampling_frequency, self.rep_seg_size, self.rep_seg_overlap_ratio)
            segments = np.transpose(segments, (2, 1, 0))  # Transpose to match shape (n_epochs, n_channels, n_times)
            
            if rm_outlier_segs:
                threshold = 6     
                outlier_counts = []
                valid_segments = np.zeros_like(segments)

                for seg_idx, segment in enumerate(segments):
                    outlier_count = 0
                    for ch_idx in range(segment.shape[0]):  # Check each channel separately
                        if is_outlier(segment[ch_idx, :], threshold):
                            outlier_count += 1
                        else:
                            valid_segments[seg_idx, ch_idx, :] = segment[ch_idx, :]
                    
                    outlier_counts.append(outlier_count)
                
                segments = valid_segments


            if not np.any(segments):  # Check if all channels are zero-padded
                print(f"All channels in window {win_idx} are zero-padded. Adding zero-padded matrix.")
                zero_padded_conn = np.zeros(conn_shape)
                conn_wpli_win.append(zero_padded_conn)
                for band_name in freq_bands_oi.keys():
                    faveraged_wpli_data = np.zeros((conn_shape[0], conn_shape[1]))
                    conn_wpli_win_favg[band_name].append(faveraged_wpli_data)
                continue

            # Calculate spectral connectivity for each segment
            conn = spectral_connectivity_epochs(
                segments,
                method=self.pli_method,
                mode='multitaper',
                mt_bandwidth=self.mt_bandwidth,
                sfreq=self.sampling_frequency,
                fmin=self.fmin,
                fmax=self.fmax,
                faverage=False, # specifies whether to average over the frequecny samples
                mt_adaptive=False,
                verbose=False
            )
            # Retrieve connectivity data and append to the list
            wpli_data = conn.get_data(output='dense')
            conn_wpli_win.append(wpli_data)

            if win_idx == 0:
                self.freqs = np.array(conn.freqs)
                # band_f_indices['bb'] = np.where((self.freqs > self.fmin_bb_avg) & (self.freqs < self.fmax_bb_avg))[0]
                for band_name, band in freq_bands_oi.items():
                    band_f_indices[band_name] = np.where((self.freqs > band[0]) & (self.freqs < band[1]))[0]

            for band_name, band in freq_bands_oi.items():
                faveraged_wpli_data = np.mean(wpli_data[:, :, band_f_indices[band_name]], axis=2)
                conn_wpli_win_favg[band_name].append(faveraged_wpli_data)

        # Convert list of arrays to a single ndarray with an additional dimension for time windows
        conn_wpli_win = np.stack(conn_wpli_win, axis=-1) if conn_wpli_win else np.array([])

        for band_name, band in freq_bands_oi.items():
            conn_wpli_win_favg[band_name] = np.stack(conn_wpli_win_favg[band_name], axis=-1) if conn_wpli_win_favg[band_name] else np.array([])

        return conn_wpli_win, conn_wpli_win_favg
    
    def calculate_pli_anatomical_average(self):
        select_chan_names = self.select_channel_names
        chan_groups = self.channel_groups

        regions = list(self.channel_groups.keys())
        self.region_names = regions
        num_regions = len(regions)
        nfres = len(self.freqs)
        freq_bands_oi_names = list(self.conn_wpli_favg[next(iter(self.conn_wpli_favg))].keys())

        def process_epoch(epoch_name=None):
            if epoch_name is None:
                num_win = self.conn_wpli_win.shape[3]
            else:
                num_win = self.conn_wpli_win[epoch_name].shape[3]

            conn_wpli_anat = np.full([num_regions, num_regions, nfres], np.nan)
            conn_wpli_favg_anat = {}
            for band_name in freq_bands_oi_names:
                conn_wpli_favg_anat[band_name] = np.full([num_regions, num_regions], np.nan)

            conn_wpli_win_anat = np.full([num_regions, num_regions, nfres, num_win], np.nan)
            conn_wpli_win_favg_anat = {}
            for band_name in freq_bands_oi_names:
                conn_wpli_win_favg_anat[band_name] = np.full([num_regions, num_regions, num_win], np.nan)

            # Retrieve connectivity data based on epoch name
            conn_wpli = self.conn_wpli if epoch_name is None else self.conn_wpli[epoch_name]
            conn_wpli_favg = self.conn_wpli_favg if epoch_name is None else self.conn_wpli_favg[epoch_name]
            conn_wpli_win = self.conn_wpli_win if epoch_name is None else self.conn_wpli_win[epoch_name]
            conn_wpli_win_favg = self.conn_wpli_win_favg if epoch_name is None else self.conn_wpli_win_favg[epoch_name]

            for region_1_idx, region_1 in enumerate(regions):
                region_1_chan_indices = np.array([select_chan_names.index(name) for name in chan_groups[region_1]])
                for region_2_idx, region_2 in enumerate(regions):
                    region_2_chan_indices = np.array([select_chan_names.index(name) for name in chan_groups[region_2]])
                    pli_idx = np.ix_(region_1_chan_indices, region_2_chan_indices)
                    
                    # Compute averages
                    conn_wpli_anat[region_1_idx, region_2_idx, :] = conn_wpli[pli_idx].mean(axis=(0, 1)).squeeze()
                    for band_name in freq_bands_oi_names:
                        conn_wpli_favg_anat[band_name][region_1_idx, region_2_idx] = conn_wpli_favg[band_name][pli_idx].mean(axis=(0, 1)).squeeze()
                    conn_wpli_win_anat[region_1_idx, region_2_idx, :, :] = conn_wpli_win[pli_idx].mean(axis=(0, 1)).squeeze() if num_win > 1 else conn_wpli_win[pli_idx].mean(axis=(0, 1))
                    for band_name in freq_bands_oi_names:
                        conn_wpli_win_favg_anat[band_name][region_1_idx, region_2_idx, :] = conn_wpli_win_favg[band_name][pli_idx].mean(axis=(0, 1)).squeeze()

            return conn_wpli_anat, conn_wpli_favg_anat, conn_wpli_win_anat, conn_wpli_win_favg_anat

        if isinstance(self.conn_wpli, dict):
            epoch_names = list(self.conn_wpli.keys())
            self.conn_wpli_anat = {}
            self.conn_wpli_favg_anat = {}
            self.conn_wpli_win_anat = {}
            self.conn_wpli_win_favg_anat = {}
            for epoch_name in epoch_names:
                conn_wpli_anat, conn_wpli_favg_anat, conn_wpli_win_anat, conn_wpli_win_favg_anat = process_epoch(epoch_name)
                self.conn_wpli_anat[epoch_name] = conn_wpli_anat
                self.conn_wpli_favg_anat[epoch_name] = conn_wpli_favg_anat
                self.conn_wpli_win_anat[epoch_name] = conn_wpli_win_anat
                self.conn_wpli_win_favg_anat[epoch_name] = conn_wpli_win_favg_anat
        else:
            self.conn_wpli_anat, self.conn_wpli_favg_anat, self.conn_wpli_win_anat, self.conn_wpli_win_favg_anat = process_epoch()