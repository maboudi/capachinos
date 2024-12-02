import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy.signal import filtfilt, spectrogram
from src.eeg_analysis.utils.helpers import create_mne_raw_from_data, select_channels_and_adjust_data
import warnings
from statsmodels.tsa.ar_model import AutoReg
from fooof import FOOOF, FOOOFGroup
from fooof.sim.gen import gen_aperiodic
from scipy.ndimage import gaussian_filter1d

class TimeFrequencyRepresentation:

    """
    Represents the time-frequency analysis of EEG data.

    This class handles the calculation, storage, manipulation, and visualization
    of time-frequency representation (TFR) data obtained from EEG recordings.
    It provides methods for plotting TFR data, as well as postprocessing through
    whitening.

    Attributes:
        data (numpy.ndarray): The TFR data, shaped as (frequencies x times x channels).
        times (numpy.ndarray): Array of time points corresponding to the second dimension of the data.
        frequencies (numpy.ndarray): Array of frequencies corresponding to the first dimension of the data.
        whitened_data (numpy.ndarray): The data after whitening transformation has been applied.
        slope (float): Parameter for the frequency-domain whitening, representing the slope of the 1/f curve.
        intercept (float): Parameter for the frequency-domain whitening, representing the intercept of the 1/f curve.

    Methods:
        plot(): Plots the TFR data using matplotlib.
        whiten(): Applies a whitening transformation to the data.
        calculate_whitened_tfr(): Returns a copy of the instance with whitened data.

    Example:
        >>> tfr = TimeFrequencyRepresentation(data=some_tfr_array, times=some_times_array, frequencies=some_frequencies_array)
        >>> tfr.plot()
        >>> whitened_tfr = tfr.calculate_whitened_tfr()
    """

    def __init__(self, data, times, frequencies):
        self.data = data
        self.times = times
        self.frequencies = frequencies

        self.reg_fit_slope = None
        self.reg_fit_intercept = None
        self.reg_fit_included_freqs = None
        self.reg_fit_aperiodic_fit = None
        
        self.fooof_periodic_params = None
        self.fooof_model = None
        self.fooof_aperiodic_fit = None 
        self.fooof_aperiodic_fit_slope = None

    def plot(self, ax=None, channel=None, start_time=0, vmin=None, vmax=None, cmap='viridis', colorbar=False, add_lables=False, title='Time-Frequency Representation'):
        
        if channel is not None:
            data_to_plot = self.data[:,:, channel]
        else:
            # Check if the data has more than one channel
            if self.data.ndim == 3 and self.data.shape[2] > 1:
                raise ValueError("Multiple channels detected. Please select a channel to plot.")
            # Otherwise, use the data as is (single channel case)
            data_to_plot = self.data
        data_to_plot[data_to_plot==0] = 1e-16
        
        if ax is None:
            _, ax = plt.subplots()
        
        adjusted_times = self.times + start_time

        Sxx_db = 10 * np.log10(np.abs(data_to_plot))

        im = ax.imshow(Sxx_db, aspect='auto', 
                       extent=[adjusted_times.min(), adjusted_times.max(), self.frequencies.min(), self.frequencies.max()], 
                       origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
        if colorbar:
            cbar = ax.figure.colorbar(im, ax=ax, aspect=10)
            cbar.set_label('Magnitude (dB)')
        if add_lables:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(title)

        return im
    
    def calculate_anatomical_group_average(self, region_name, ch_names, channel_groups):
        """
        Calculate average time frequency representation for a predefined anatomical channel group.
        
        Parameters:
            ch_names (list): List of channel names corresponding to the third dimension of tfr.data
        """
        tfr_average = TimeFrequencyRepresentation(np.copy(self.data), self.times, self.frequencies)
        
        # Find the indices fo the channels that belong to the current anatomical region
        region_channel_indices = [i for i, ch in enumerate(ch_names) if ch in channel_groups[region_name]]
        
        region_data = self.data[:, :, region_channel_indices]               
        tfr_average.data = np.nanmean(region_data, axis=2)
    
        return tfr_average

    def calculate_whitened_tfr(self, reg_ap_fit, averaging_win_size=None, averaging_step_size=None, fbands_info=None):
        """
        Create a whitened copy of the TFR data and returns it.
        """
        tfr_whitened = TimeFrequencyRepresentation(np.copy(self.data), self.times, self.frequencies)
        tfr_whitened.whiten(reg_ap_fit, averaging_win_size, averaging_step_size, fbands_info)

        return tfr_whitened

    # def whiten(self, baseline_data=None, fooof_periodic_params=None):
    #     """
    #     Apply whitening to the TFR data using a baseline or the entire dataset
    #     to calculate the whitening parameters.

    #     - baseline_data: Optional; a 3D numpy array of TFR data from a baseline epoch, 
    #                      or a concatenation of multiple epochs if provided.
    #     This method alters the TFR data in place.

    #     It sets `self.whitened_data` with the whitened TFR data.
    #     """
    #     self._calculate_whitening_parameters(baseline_data, fooof_periodic_params)
        
    #     freqs = self.frequencies[:, np.newaxis]
        
    #     tfr_data_db = 10 * np.log10(np.abs(self.data))
    #     tfr_db_whitened = tfr_data_db - (self.slope * freqs + self.intercept)
    #     self.data = 10 ** (tfr_db_whitened / 10)

    def whiten(self, reg_ap_fit=None, averaging_win_size=None, averaging_step_size=None, fbands_info=None):
        """
        Apply whitening to the TFR data by subtracting aperiodic fit(s) from the original TFR data.

        Parameters
        ----------
        reg_ap_fit : array-like
            The aperiodic fit(s) to be subtracted from the original TFR data.
            If the fit is calculated based on the channel-averaged TFRs (for example, in region-averaged TFRs),
            it could be based on the baseline or the same epoch on which the whitening is applied.
        averaging_win_size : int
            The size of each window in samples.
        averaging_step_size : int
            The step size of the moving windows in samples.
        fbands_info : dict, optional
            Information on the frequency bands for which the power relative to aperiodic fit is calculated.
            
        Returns
        -------
        None
            The function updates the `self.data` attribute with the whitened TFR data and sets `self.whitened_data`
            to store the whitened TFR data.

        >>> NOTE: In the case of overlapping moving windows, it is necessary to convert the moving windows and the corresponding
        aperiodic fits to non-overlapping windows because whitening needs to be done in non-overlapping time windows.
        """
        # Validate inputs
        if reg_ap_fit is None or averaging_win_size is None or averaging_step_size is None:
            raise ValueError("Missing required inputs: reg_ap_fit, averaging_win_size, or averaging_step_size")

        def apply_whitening(tfr_db_whitened, tfr_data_db, reg_ap_fit, fit_num_wins, tfr_times, avg_win_size):
            """
            Helper function to apply whitening based on the aperiodic fit.
            """
            for win_idx in range(fit_num_wins):
                start_idx = win_idx * avg_win_size
                end_idx = (win_idx + 1) * avg_win_size
                curr_win_time_pnt_idx = (tfr_times >= start_idx) & (tfr_times < end_idx)
                curr_win_num_time_pnts = np.sum(curr_win_time_pnt_idx)

                tfr_num_channels = tfr_data_db.shape[2]

                curr_win_reg_ap_fit = reg_ap_fit[:, win_idx, :]
                if curr_win_reg_ap_fit.ndim == 2:
                    curr_win_reg_ap_fit = curr_win_reg_ap_fit[:, :, np.newaxis]
                curr_win_reg_ap_fit_broadcasted = np.tile(curr_win_reg_ap_fit, (1, curr_win_num_time_pnts, tfr_num_channels))
                tfr_db_whitened[:, curr_win_time_pnt_idx, :] = tfr_data_db[:, curr_win_time_pnt_idx, :] - curr_win_reg_ap_fit_broadcasted

            return tfr_db_whitened

        tfr_data_db = 10 * np.log10(np.abs(self.data))

        if tfr_data_db.ndim < 3:
            tfr_data_db = tfr_data_db[:, :, np.newaxis]

        num_freqs, num_time_pnts, num_channels = tfr_data_db.shape

        # extract the the aperiodic fits corresponding to non-overlapping windows
        step_factor = averaging_win_size // averaging_step_size
        if reg_ap_fit.shape[1] > 1:
            reg_ap_fit = reg_ap_fit[:, ::step_factor, :]

        _, fit_num_wins, fit_num_channels = reg_ap_fit.shape

        tfr_db_whitened = np.full((num_freqs, num_time_pnts, num_channels), np.nan)
            
        if fit_num_channels == 1: # when the fit is calculated based on channel_averaged TFRs like in region_avg TFR. 
            # The fit calculation could be based on the data during the baseline or the epoch that the whitening is applied on. The data on which the whitening is applied on can be chan averaged or not. 
            if fit_num_wins == 1: # when baseline is used for fit calculation
                curr_reg_ap_fit = reg_ap_fit[:, 0, 0]
                reg_ap_fit_broadcasted = np.tile(curr_reg_ap_fit[:, np.newaxis, np.newaxis], (1, num_time_pnts, num_channels))
                tfr_db_whitened = tfr_data_db - reg_ap_fit_broadcasted            

            else: # when the aperiodic fit is calculated using the same data that is whitening is applied on
                apply_whitening(tfr_db_whitened, tfr_data_db, reg_ap_fit, fit_num_wins, self.times, averaging_win_size)      
        else:
            if fit_num_channels != num_channels:
                raise ValueError("Inconsistency in the number of channels between the TFR data and the aperiodic_fit array")
            
            if fit_num_wins == 1: # fit based on baseline 
                for chan_idx in range(num_channels):
                    curr_reg_ap_fit = reg_ap_fit[:, 0, chan_idx]
                    reg_ap_fit_broadcasted = np.tile(curr_reg_ap_fit[:, np.newaxis, np.newaxis], (1, num_time_pnts, num_channels))
                    tfr_db_whitened[:, :, chan_idx] = tfr_data_db[:, :, chan_idx] - reg_ap_fit_broadcasted 

            else: # aperiodic fit is based on the same data
                for chan_idx in range(num_channels):
                    tfr_db_whitened[:, :, chan_idx] = apply_whitening(tfr_db_whitened, tfr_data_db, reg_ap_fit[:, :, chan_idx], fit_num_wins, self.times, averaging_win_size)
        
        self.data = 10 ** (tfr_db_whitened / 10)
        self.whitened_data = self.data  # Assuming `self.whitened_data` needs to store the whitened data

    def calculate_whitening_parameters(self, baseline_data=None, per_channel=False, exclude_fbands=True, peak_threshold=0.1, window_size=None, step_size=None, fbands=None, freq_range=None, fooof_aperidoc_freq_start=None):
        """
        Calculates whitening parameters based on the provided baseline data or the current instance data.

        Parameters:
        baseline_data (Optional[np.ndarray]): Baseline data to calculate whitening parameters from.
        fooof_periodic_params (Optional[Dict]): Information about frequency bands to exclude.
        per_channel (bool): Whether to calculate separate slopes and intercepts for each channel if baseline data is provided or per window.
        freq_range (List[int]): Frequency range for fitting linear models, default is [7, 35] Hz.

        excluded_fbands are only used when no baseline_data is specified. 
        If baseline_data is provided, the FOOOF parameters and therefore excluded_fbands are calculated for the average baseline power spectral density.   

        """
        freqs = self.frequencies
        num_frequencies = len(freqs)

        if freq_range is None:
            freq_range = [7, 35]

        def calculate_included_freqs(excluded_fbands, freqs, freq_range, curr_chan_idx = None, current_win_idx=None):
            if curr_chan_idx is None:
                curr_chan_idx = 0
            if current_win_idx is None:
                current_win_idx = 0

            included_freqs_idx = np.full((len(freqs), ), True)
            for fband, band_params in excluded_fbands.items():
                lower_bound = 0 if fband == 'delta' else band_params['lower_bound'][curr_chan_idx, current_win_idx]
                upper_bound = 4 if fband == 'delta' else band_params['upper_bound'][curr_chan_idx, current_win_idx]

                if not np.isnan(lower_bound) and not np.isnan(upper_bound):
                    included_freqs_idx *= ((freqs <= lower_bound) | (freqs >= upper_bound)) & (freqs > freq_range[0]) & (freqs < freq_range[1])

            return included_freqs_idx
        

        included_freqs_idx = np.full((len(freqs), ), True)

        if baseline_data is not None: 

            if baseline_data.dim == 3 and per_channel:

                num_channels = baseline_data.shape[2]
                num_wins = 1  # Since we're averaging over windows

                self.reg_fit_slope = np.full((num_wins, num_channels), np.nan)
                self.reg_fit_intercept = np.full((num_wins, num_channels), np.nan)
                self.reg_fit_included_freqs = np.full((num_frequencies, num_wins, num_channels), np.nan, dtype=bool)
                self.reg_fit_aperiodic_fit = np.full((num_frequencies, num_wins, num_channels), np.nan)

                # averge over the time axis following removel of outlier time points
                mean_spectral_power, _ = TimeFrequencyRepresentation._calculate_mean_spectral_power_with_outlier_removal(
                    baseline_data, axis=1
                )

                for chan_idx in range(num_channels):
                    curr_chan_mean_spectral_power = mean_spectral_power[:, :, chan_idx]

                    psd_db = 10 * np.log10(curr_chan_mean_spectral_power)

                    if exclude_fbands: 
                        # Calculate the FOOOF periodic parameters
                        # NOTE: no need to return the FOOOF aperiodic component and the drived slope from the baseline data
                        fooof_periodic_params, _, _, _ = self.calculate_periodic_parameters(
                            tfr_data=curr_chan_mean_spectral_power, fbands=fbands, peak_threshold=peak_threshold, overlap_threshold=0.5
                        )
                        included_freqs_idx = calculate_included_freqs(fooof_periodic_params, freqs, freq_range=freq_range)
                        
                    # Fit a linear model (in log power-linear frequency space) to calculate slope and intercept
                    self.reg_fit_slope[0, chan_idx], self.reg_fit_intercept[0, chan_idx] = np.polyfit(
                        freqs[included_freqs_idx], psd_db[included_freqs_idx], 1
                    )
                    self.reg_fit_included_freqs[:, 0, chan_idx] = included_freqs_idx
                    self.reg_fit_aperiodic_fit[:, 0, chan_idx] = (
                        self.reg_fit_intercept[0, chan_idx] + self.reg_fit_slope[0, chan_idx] * freqs
                    )

            else:
                num_channels = 1
                num_wins = 1  # Since we're averaging over windows

                self.reg_fit_slope = np.full((num_wins, num_channels), np.nan)
                self.reg_fit_intercept = np.full((num_wins, num_channels), np.nan)
                self.reg_fit_included_freqs = np.full((num_frequencies, num_wins, num_channels), np.nan, dtype=bool)
                self.reg_fit_aperiodic_fit = np.full((num_frequencies, num_wins, num_channels), np.nan)

                mean_spectral_power, _ = TimeFrequencyRepresentation._calculate_mean_spectral_power_with_outlier_removal(
                    baseline_data   
                )

                psd_db = 10 * np.log10(mean_spectral_power)

                if exclude_fbands:
                    fooof_periodic_params, _, _, _= self.calculate_periodic_parameters(
                        tfr_data=mean_spectral_power, fbands=fbands, peak_threshold=peak_threshold, overlap_threshold=0.5
                    )
                    included_freqs_idx = calculate_included_freqs(fooof_periodic_params, freqs, freq_range=freq_range) 
                
                self.reg_fit_slope[0, 0], self.reg_fit_intercept[0, 0] = np.polyfit(
                    freqs[included_freqs_idx], psd_db[included_freqs_idx], 1
                )
                self.reg_fit_included_freqs[:, 0, 0] = included_freqs_idx
                self.reg_fit_aperiodic_fit[:, 0, 0] = (
                    self.reg_fit_intercept[0, 0] + self.reg_fit_slope[0, 0] * freqs
                )

        else: # no baseline data for calculation of slope. The regression line parameters are calculated for each individual window and used later for whitening or other procedures
            
            if window_size is None:
                raise ValueError('Window size is missing for calculation of FOOOF periodic parameters')

            tfr_windowed, _, _ = self.calculate_window_average(window_size, step_size)
            _, num_wins, num_channels = tfr_windowed.shape

            self.reg_fit_slope = np.full((num_wins, num_channels), np.nan)
            self.reg_fit_intercept = np.full((num_wins, num_channels), np.nan)
            self.reg_fit_included_freqs = np.full((num_frequencies, num_wins, num_channels), np.nan, dtype=bool)
            self.reg_fit_aperiodic_fit = np.full((num_frequencies, num_wins, num_channels), np.nan)

            if self.data.ndim == 3 and per_channel:

                if exclude_fbands: 
                    self.fooof_periodic_params, self.fooof_model, self.fooof_aperiodic_fit, self.fooof_aperiodic_fit_slope = self.calculate_periodic_parameters(
                        tfr_data=tfr_windowed, fbands=fbands, peak_threshold=peak_threshold, overlap_threshold=0.5, aperiodic_fit_start_frequency=fooof_aperidoc_freq_start
                    )

                for chan_idx in range(num_channels):
                    for win_idx in range(num_wins):
                        psd_db = 10 * np.log10(tfr_windowed[:, win_idx ,chan_idx])

                        if exclude_fbands:
                            included_freqs_idx = calculate_included_freqs(
                                self.fooof_periodic_params, freqs, freq_range=freq_range, curr_chan_idx=chan_idx, current_win_idx=win_idx
                            )
                        
                        self.reg_fit_slope[win_idx, chan_idx], self.reg_fit_intercept[win_idx, chan_idx] = np.polyfit(
                            freqs[included_freqs_idx], psd_db[included_freqs_idx], 1
                        )
                        self.reg_fit_included_freqs[:, win_idx, chan_idx] = included_freqs_idx
                        self.reg_fit_aperiodic_fit[:, win_idx, chan_idx] = (
                            self.reg_fit_intercept[win_idx, chan_idx] + self.reg_fit_slope[win_idx, chan_idx] * freqs
                        )
            else:
                num_channels = 1

                # take an average of the windowed tfr data over all channels 
                mean_spectral_power, _ = TimeFrequencyRepresentation._calculate_mean_spectral_power_with_outlier_removal(
                    tfr_windowed, axis=2, remove_outliers=False
                )

                if exclude_fbands:
                    self.fooof_periodic_params, self.fooof_model, self.fooof_aperiodic_fit, self.fooof_aperiodic_fit_slope = self.calculate_periodic_parameters(
                        tfr_data=mean_spectral_power, fbands=fbands, peak_threshold=peak_threshold, overlap_threshold=0.5, aperiodic_fit_start_frequency=fooof_aperidoc_freq_start
                    )

                for win_idx in range(num_wins):
                    curr_win_mean_spectal_power = mean_spectral_power[:, win_idx]
                    psd_db = 10 * np.log10(curr_win_mean_spectal_power)

                    if exclude_fbands:
                        included_freqs_idx = calculate_included_freqs(
                            self.fooof_periodic_params, freqs, freq_range=freq_range, current_win_idx=win_idx
                        )
                    
                    self.reg_fit_slope[win_idx, 0], self.reg_fit_intercept[win_idx, 0] = np.polyfit(
                        freqs[included_freqs_idx], psd_db[included_freqs_idx], 1
                    )
                    self.reg_fit_included_freqs[:, win_idx, 0] = included_freqs_idx
                    self.reg_fit_aperiodic_fit[:, win_idx, 0] = (
                        self.reg_fit_intercept[win_idx, 0] + self.reg_fit_slope[win_idx, 0] * freqs
                    )
                

    # def _calculate_whitening_parameters(self, baseline_data=None):
    #     """
    #     Calculates whitening parameters based on the provided baseline data or the current instance data.
    #     """
    #     if baseline_data is None:
    #         baseline_data = self.data

    #     freqs = self.frequencies

    #     # Identify non-outlier time points and calculate mean spectral power
    #     mean_spectral_power, _ = TimeFrequencyRepresentation._calculate_mean_spectral_power_with_outlier_removal(
    #         baseline_data
    #     )
    #     tfr_db = 10 * np.log10(mean_spectral_power)

    #     # Fit a linear model (in log-log space) to calculate slope and intercept
    #     # log_freqs = np.log10(freqs)
    #     # slope, intercept = np.polyfit(log_freqs, tfr_db, 1)
    #     slope, intercept = np.polyfit(freqs, tfr_db, 1)

    #     self.slope = slope
    #     self.intercept = intercept
    
    def calculate_window_average(self, window_size, step_size=None, smoothing_sigma=3):
        """
        Calculate average TFRs within each window

        Parameters:
        window_size (float): Duration of the window in seconds.
        step_size (float): advance of the window start time on each iteration in seconds.

        Returns:
        tfr_mean (np.ndarray): Mean spectral power for each channel within each window
        tfr_std (np.ndarray): standard deviation of spectral power for each channel within each window
        window_centers (np.ndarray): array of center times for each window
        """ 
        if step_size is None:
            step_size = window_size

        tfr_data = self.data
        if tfr_data.ndim == 2: 
            tfr_data = tfr_data[..., np.newaxis]

        num_freqs, _, num_channels = tfr_data.shape
        tfr_times = self.times

        T = tfr_times[-1] - tfr_times[0]
        num_windows = 1 + int((T - window_size)/step_size) if T > window_size else 1

        window_starts = [tfr_times[0]+i * step_size for i in range(num_windows)]
        window_centers = [start + window_size/2 for start in window_starts]

        tfr_mean = np.full((num_freqs, num_windows, num_channels), np.nan)
        tfr_std = np.full((num_freqs, num_windows, num_channels), np.nan)

        for win_idx, start in enumerate(window_starts):
            end = min(start + window_size, tfr_times[-1])
            in_window = (tfr_times >= start) & (tfr_times < end)
            for chn_idx in range(num_channels):
                segment = tfr_data[:, in_window, chn_idx]
                mean_power, std_power = TimeFrequencyRepresentation._calculate_mean_spectral_power_with_outlier_removal(segment)
                
                # Smoothing the mean_spectrum 
                mean_power = gaussian_filter1d(mean_power, smoothing_sigma)
                
                tfr_mean[:, win_idx, chn_idx] = mean_power
                tfr_std[:, win_idx, chn_idx] = std_power

        return tfr_mean, tfr_std, np.array(window_centers), 

    def calculate_aperiodic_parameters(self, freq_range=None, aperiodic_mode='fixed'):
        """
        Calcualte the aperiodic components in the defined freq_range.
        For calcualtion of the aperiodic parameters, the linear section of the spectrum,
        likely within the default freq_range is considered and the 'fixed' aperiodic mode is used,
        because the interpretation of the calculated exponent and intercept is more straightforward 
        witohut a "knee" parameter being calculated.  

        Parameters:
        freq_range (list): the frequency range within which the FOOOF method is applied to the spectrum.
        aperiodic_mode (str): The aperiodic mode used in FOOOF 
        """
        
        if freq_range is None:
            freq_range = [20, 40]

        tfr_data = self.data
        if tfr_data.ndim == 2: 
            tfr_data = tfr_data[..., np.newaxis]
        
        _, num_time_points, num_channels = tfr_data.shape
        tfr_freqs = self.frequencies

        exponents = np.full((num_channels, num_time_points), np.nan)
        intercepts = np.full((num_channels, num_time_points), np.nan)
        knees = np.full((num_channels, num_time_points), np.nan)

        for chn_idx in range(num_channels):
            fooof_model = FOOOFGroup(aperiodic_mode=aperiodic_mode, verbose=False)
            try:
                fooof_model.fit(tfr_freqs, tfr_data[:, :, chn_idx].T, freq_range=freq_range)
            except:
                array = tfr_data[:, :, chn_idx].T.flatten()
                rr = np.where(~np.isnan(array))[0]
                continue

            parameters = fooof_model.get_params('aperiodic_params')

            exponents[chn_idx, :] = parameters[:, 1]
            intercepts[chn_idx, :] = parameters[:, 0]
            knees[chn_idx, :] = parameters[:, 2] if aperiodic_mode=='knee' else None

        fooof_aperiodic_parameters = {
            'exponents': exponents,
            'intercepts': intercepts
        }

        if aperiodic_mode == 'knee':
            fooof_aperiodic_parameters['knees'] = knees

        return fooof_aperiodic_parameters
    
    def calculate_periodic_parameters(self, tfr_data, freq_range=None, fbands=None, aperiodic_mode='knee', peak_threshold=1.5, overlap_threshold=0.25, aperiodic_fit_start_frequency=None):
        """
        Calculate the periodic paramters given the bands.
        The method utilizes FOOOF for fitting peak within the bands, after subtracting the 
        aperiodic component. The aperiodic compnent estimated using knee is more accurate when
        the frequency range is wide, such as the whole signal frequency range.

        Parameters:
        bands:
        aperiodic_mode:
        """
        if freq_range is None:
            freq_range = [0.5, self.frequencies[-1]]

        if aperiodic_fit_start_frequency is None:
            aperiodic_fit_start_frequency = 10
        
        # if fbands is None:
        #     fbands = {
        #         'delta': [0.5, 4],
        #         'theta': [4, 7],
        #         'alpha':[7, 14],
        #         'beta': [18, 35],
        #         'gamma':[35, 55]
        #     }
        #     warnings.warn("No frequency band was specified to initialize calculation of FOOOF aperiodic parametes, so default bands will be used")
        

        # Sort the bands based on the first frequency
        fbands = dict(sorted(fbands.items(), key=lambda item: item[1][0]))

        # tfr_data = self.data
        if tfr_data.ndim == 2: 
            tfr_data = tfr_data[..., np.newaxis]
        
        _, num_time_points, num_channels = tfr_data.shape
        tfr_freqs = self.frequencies

        # Initialize a structure to store FOOOF models only if the conditions are met
        fooof_model = {}
        # if num_time_points <= 50 and num_channels == 1:
        #     fooof_model = {}
        # else:
        #     warnings.warn(f"Not storing the fooof_model due to the number of spectrums given the number of time windows ({num_time_points}) and channels ({num_channels}).")
        #     fooof_model = None
        
         # Initialize a structure to store the periodic parameters for all channels, time points, and bands.
        nan_array = np.full((num_channels, num_time_points), np.nan)
        fooof_periodic_params = {
            fband: {
                key: nan_array.copy() for key in (
                    'cf', 
                    'amplitude', 
                    'bw', 
                    'lower_bound', 
                    'upper_bound', 
                    'avg_power_absolute', 
                    'avg_power_relative'
                    )
            }
            for fband in fbands
        }
        aperiodic_fit = {}
        aperiodic_fit_slope = nan_array.copy()

        for chn_idx in range(num_channels):
            for time_pt in range(num_time_points):
                # Fit FOOOF model
                curr_fooof_model = FOOOF(
                    aperiodic_mode=aperiodic_mode, peak_threshold=peak_threshold, verbose=False
                )

                try:
                    curr_fooof_model.fit(
                        tfr_freqs, tfr_data[:, time_pt, chn_idx], freq_range=freq_range
                    )

                except:
                    print(f'Error fitting FOOOF model')
                    continue

                if curr_fooof_model is not None:
                    fooof_model[(chn_idx, time_pt)] = curr_fooof_model
                
                    # aperiodic_params = fooof_model.get_params('aperiodic_params')
                    # aperiodic_fit = self._aperiodic_fit(tfr_freqs, aperiodic_params)
                    # aperiodic_fit = fooof_model._ap_fit
                    # try:
                    curr_aperiodic_fit = gen_aperiodic(
                        curr_fooof_model.freqs, curr_fooof_model._robust_ap_fit(curr_fooof_model.freqs, curr_fooof_model.power_spectrum)
                    )

                    # Loop over bands and store the peak parameters
                    prev_upper_bound = np.nan
                    prev_band_label = None
                    for band_label, freq_band in fbands.items():
                        if band_label != 'delta':
                            if band_label in ['beta']:
                                bandwidth_factor = 1/1.5
                            elif band_label in ['gamma']:
                                bandwidth_factor = 1/6
                            # elif band_label == 'theta':
                            #     bandwidth_factor = 1/20
                            else:
                                bandwidth_factor = 1/4

                            extracted_peak = self._extract_highest_peak_in_band(
                                curr_fooof_model, freq_band, bandwidth_factor, overlap_threshold=overlap_threshold
                            )
                            
                            if ~np.all(np.isnan(np.array(extracted_peak))):         
                                fooof_periodic_params[band_label]['cf'][chn_idx, time_pt] = extracted_peak[0][0]
                                fooof_periodic_params[band_label]['amplitude'][chn_idx, time_pt] = extracted_peak[0][1]
                                fooof_periodic_params[band_label]['bw'][chn_idx, time_pt] = extracted_peak[0][2]
                                lower_bound = extracted_peak[0][3]
                                upper_bound = extracted_peak[0][4]
                            else:
                                upper_bound = np.nan
                                lower_bound = np.nan
                                # print(f"No peak found in band {band_label} for channel {chn_idx} at time {time_pt}.")
                        
                            fooof_periodic_params[band_label]['upper_bound'][chn_idx, time_pt] = upper_bound

                            if (~np.isnan(prev_upper_bound)) and (prev_band_label is not None):
                                if lower_bound <= prev_upper_bound:
                                    if band_label in ['beta']:
                                        fooof_periodic_params[band_label]['lower_bound'][chn_idx, time_pt] = prev_upper_bound + 1
                                    else:
                                        middle_value = (lower_bound + prev_upper_bound)/2
                                        lower_bound = middle_value
                                        fooof_periodic_params[band_label]['lower_bound'][chn_idx, time_pt] = middle_value + 0.5
                                        fooof_periodic_params[prev_band_label]['upper_bound'][chn_idx, time_pt] = middle_value - 0.5
                                else:
                                    fooof_periodic_params[band_label]['lower_bound'][chn_idx, time_pt] = lower_bound
                            else:
                                fooof_periodic_params[band_label]['lower_bound'][chn_idx, time_pt] = lower_bound

                            if ~np.isnan(upper_bound):
                                prev_upper_bound = upper_bound
                                prev_band_label = band_label

                    # Calculate absolute and relative (to the aperiodic component) power
                    for band_label in fooof_periodic_params.keys():

                        if band_label != 'delta':
                            lower_bound = fooof_periodic_params[band_label]['lower_bound'][chn_idx, time_pt]
                            upper_bound = fooof_periodic_params[band_label]['upper_bound'][chn_idx, time_pt]
                        else:
                            lower_bound, upper_bound = fbands[band_label]

                        if lower_bound is not None and upper_bound is not None:
                            band_mask = (tfr_freqs >= lower_bound) & (tfr_freqs <= upper_bound)

                            if band_mask.any():
                                power_absolute = np.mean(curr_fooof_model.power_spectrum[band_mask])
                                power_relative = np.mean(curr_fooof_model.power_spectrum[band_mask] - curr_aperiodic_fit[band_mask])
                            else:
                                power_absolute = np.nan
                                power_relative = np.nan
                            
                            fooof_periodic_params[band_label]['avg_power_absolute'][chn_idx, time_pt] = power_absolute
                            fooof_periodic_params[band_label]['avg_power_relative'][chn_idx, time_pt] = power_relative

                    aperiodic_fit[(chn_idx, time_pt)] = curr_aperiodic_fit

                    freq_includ_idx = np.where(self.frequencies > aperiodic_fit_start_frequency)[0]
                    slope, _ = np.polyfit(self.frequencies[freq_includ_idx], curr_aperiodic_fit[freq_includ_idx], 1)
                    aperiodic_fit_slope[chn_idx, time_pt] = -10*slope # since aperiodc_fit is of the type log(power). So, to calculate in dB we need to multiply by 10

                    # except:
                    #     print('Faining to process FOOOF model for the current window/time point')

        return fooof_periodic_params, fooof_model, aperiodic_fit, aperiodic_fit_slope

    # def _aperiodic_fit(self, freqs, params):
    #     """
    #     Calculate the aperiodic fit across the given frequencies.

    #     Args:
    #         freqs (ndarray): The frequency values to calculate the aperiodic fit for.
    #         params (list or ndarray): The aperiodic parameters from the FOOOF model.

    #     Returns:
    #         ndarray: The aperiodic fit for the given frequencies.
    #     """
    #     if len(params) == 2:  # No knee, simple 1/f model
    #         # params[0] is the offset, params[1] is the exponent
    #         return params[0] - params[1] * np.log10(freqs)
    #     elif len(params) == 3:  # Knee fit
    #         # params[0] is the offset, params[1] is the knee, params[2] is the exponent
    #         return params[0] - params[2]*(np.log10((freqs**2) + (params[1]**2)))

    def _combine_overlapping_bandwidths(self, peaks, overlap_threshold):
        """
        Combines bandwidths of peaks that overlap by a significant amount.

        Parameters:
          peaks (list): A list of peak parameters, where each peak is a tuple (CF, Amp, BW).
          overlap_threshold (float): The fraction of overlap between bandwidths required to consider peaks overlapping.

        Returns:
          tuple: A tuple containing the combined peak parameters (CF, Amp, combined BW).
        """
        # Assume peaks are sorted by amplitude, highest first
        # Each peak is a (center_frequency, amplitude, bandwidth) tuple
        highest_peak = peaks[0]
        combine_bw_start = highest_peak[0] - (highest_peak[2] / 2)  # start frequency of highest peak's bandwidth
        combine_bw_end = highest_peak[0] + (highest_peak[2] / 2)    # end frequency of highest peak's bandwidth

        # Function to calculate overlap percentage
        def calculate_overlap_percentage(bw1_start, bw1_end, bw2_start, bw2_end):
            overlap_start = max(bw1_start, bw2_start)
            overlap_end = min(bw1_end, bw2_end)
            overlap = max(0, overlap_end - overlap_start)
            smallest_bw = min(bw1_end - bw1_start, bw2_end - bw2_start)
            return overlap / smallest_bw if smallest_bw > 0 else 0

        # Check and merge with overlapping peaks
        for peak in peaks[1:]:
            # Calculate the bandwidth range of the current peak
            bw_start = peak[0] - (peak[2] / 2)
            bw_end = peak[0] + (peak[2] / 2)

            # Determine overlap by seeing if bandwidths intersect
            overlap_percentage = calculate_overlap_percentage(
                combine_bw_start, combine_bw_end, bw_start, bw_end
            )

            if overlap_percentage >= overlap_threshold:
                # Update combined bandwidth range
                combine_bw_start = min(combine_bw_start, bw_start)
                combine_bw_end = max(combine_bw_end, bw_end)

        combine_bw = combine_bw_end - combine_bw_start
        return (highest_peak[0], highest_peak[1], combine_bw, combine_bw_start, combine_bw_end)
    
    def _extract_highest_peak_in_band(self, fooof_model, freq_band, band_width_factor, overlap_threshold):
        """
        Extracts the highest peak within a specified frequency band from FOOOF results.

        Parameters:
          fooof_model (FOOOFGroup): The fitted FOOOFGroup object.
          freq_band (tuple): The frequency band of interest.

        Returns:
          list: A list of the highest peak parameters with combined bandwidths for the specified frequency band.
        """
        highest_peaks_with_overlaps = []

        peaks = fooof_model.get_params('peak_params')
        if peaks.ndim > 1 and peaks.shape[0] > 0:
            for peak_idx in range(peaks.shape[0]):
                # Calculate the standard deviation based on the bandwidth (FWHM)
                gaussian_kernel_sd = peaks[peak_idx, 2]/(2*np.sqrt(2 * np.log(2)))
                
                # Full bandwidth is ±3 SD
                X = band_width_factor
                full_bw = 2. * gaussian_kernel_sd * np.sqrt(-2 * np.log(X)) 
                
                # full_bw = 2*1.96*gaussian_kernel_sd

                # Assign the full width to the bandwidth column
                peaks[peak_idx, 2] = full_bw
            band_peaks = [peak for peak in peaks if freq_band[0] <= peak[0] <= freq_band[1]]
        else:
            band_peaks = []

        # If we did get band peaks, sort them and proceed.
        if band_peaks:
            # Sort peaks based on amplitude
            band_peaks.sort(key=lambda peak: peak[1], reverse=True)
            # Combine bandwidths of overlapping peaks and get highest combined peak
            highest_peak_with_overlap = self._combine_overlapping_bandwidths(band_peaks, overlap_threshold)
        else:
            highest_peak_with_overlap = (np.nan,) * 5 
        
        highest_peaks_with_overlaps.append(highest_peak_with_overlap)

        return highest_peaks_with_overlaps

    @staticmethod
    def _calculate_mean_spectral_power_with_outlier_removal(tfr_data, axis=None, remove_outliers=True):
        """
        Identifies outlier time points and calculates the mean spectral power,
        excluding those outliers.

        axes: axis along which the avaeraging is applied
        """
        if tfr_data.ndim < 3: #if processing channel average data
            tfr_data = tfr_data[..., np.newaxis]

        if axis is None:
            axis = (1, 2)

        tfr_magnitude = np.abs(tfr_data)
        sum_over_freq_and_channels = np.sum(tfr_magnitude, axis=(0, 2))

        if remove_outliers:
            non_outlier_time_indices = TimeFrequencyRepresentation._identify_non_outlier_time_indices(sum_over_freq_and_channels)
            tfr_filtered = tfr_magnitude[:, non_outlier_time_indices, :]
        else:
            tfr_filtered = tfr_magnitude

        mean_spectral_power = np.mean(tfr_filtered, axis=axis)
        std_spectral_power = np.std(tfr_filtered, axis=axis)

        return mean_spectral_power, std_spectral_power
    
    @staticmethod
    def _identify_non_outlier_time_indices(summed_data):
        """
        Identifies and returns indices of non-outliers based on the summed data.
        """
        q1 = np.percentile(summed_data, 25)
        q3 = np.percentile(summed_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        non_outlier_indices = np.where((summed_data >= lower_bound) & (summed_data <= upper_bound))[0]
        return non_outlier_indices


class PowerSpectralAnalysis:
    """
    A class for performing power spectral analysis on EEG data.

    This class provides a comprehensive suite of methods for analyzing EEG
    data in both the time and frequency domains. It supports various time-frequency
    transformation methods and allows for preprocessing such as time domain
    whitening and postprocessing such as frequency domain whitening. The class
    can handle single or multiple EEG epochs.

    Attributes:
        participant_id (str): Identifier for the participant whose EEG data is analyzed.
        eeg_data (numpy.ndarray or dict): Preprocessed EEG data. Can be a NumPy array
            for continuous EEG or a dictionary with different epochs as keys in epoched EEG.
        time_domain_whitened (bool): Flag indicating if time domain whitening was applied.
        channel_names (list): Names of the EEG channels.
        sampling_frequency (float): Sampling frequency of the EEG data.
        freq_resolution (float): Frequency resolution for spectral analysis.
        frequencies (numpy.ndarray): Array of frequencies at which to compute the power
            spectral density.
        nperseg (int): Number of samples per segment for spectral decomposition.
        noverlap (int): Number of overlapping samples per segment.
        time_step (float): Time step between successive samples after decimation.
        tfr (dict): Dictionary containing TimeFrequencyRepresentation instances for each epoch.
        whitened_tfr (dict): Dictionary containing whitened TimeFrequencyRepresentation instances.
        
    Methods:
        preprocess_time_domain_whitening(): Apply time domain whitening on the raw EEG data.
        calculate_time_frequency_map(): Perform time-frequency transformation on EEG data.
        postprocess_frequency_domain_whitening(): Apply frequency domain whitening on the time-frequency transformed data.
        
    Example:
        >>> psa = PowerSpectralAnalysis(eeg_file)
        >>> psa.preprocess_time_domain_whitening()
        >>> psa.calculate_time_frequency_map(method='multitaper', select_channels=['Pz', 'Oz'])
        >>> psa.postprocess_frequency_domain_whitening(baseline_epoch_name='preop_rest')
    """

    def __init__(self, 
                 eeg_file, 
                 window_size=None, 
                 step_size=None):
        """
        Initialize the PowerSpectralAnalysis with (preprocessed) EEG data.
        """
        self.participant_id = eeg_file.participant_id
        self.eeg_data = eeg_file.eeg_data
        self.channel_names = eeg_file.channel_names
        self.channel_groups = eeg_file.channel_groups
        self.sampling_frequency = eeg_file.sampling_frequency

        # self.select_channels = None
        self.select_channel_names = None

        self.whitened_eeg = None
        self.time_domain_whitened = False
              
        self.freq_resolution = 0.5
        self.frequencies = np.arange(0.5, 55.5, self.freq_resolution) 

        window_size_samples = window_size*self.sampling_frequency if window_size else int(self.sampling_frequency)
        step_size_samples = step_size*self.sampling_frequency if step_size else window_size_samples // 2
        self.nperseg = window_size_samples
        self.time_step = step_size
        self.noverlap = (window_size_samples - step_size_samples)/self.sampling_frequency

        self.tfr = {}
        
        self.whitened_tfr = {}
        self.whitened_tfr = {}
        self.regression_ap_fit = {}
        self.regresssion_ap_fit_slope = {}
        self.regression_ap_fit_intercept = {}
        self.regression_ap_fit_included_freqs = {}

        self.region_average_tfr = {}
        self.window_average_tfr = {}

        self.fooof_aperiodic_parameters = {}
        self.fooof_periodic_parameters = {}
        self.fooof_model = {}
        self.fooof_aperiodic_fit = {}
        self.fooof_aperiodic_fit_slope = {}

        self.fband_power_relative_reg_ap = {}

    def preprocess_time_domain_whitening(self, AR_order=2, baseline_epoch_name=None, concatenate_data=False, window_size=None, coeff_mode='average'):
        """
        Apply time domain whitening on the EEG data using an AutoRegressive model.
        The AR model coefficients can be calculated based on various mode settings.
        
        Parameters:
        - AR_order: The order of the autoregressive model to use for calculating whitening parameters.
        - baseline_epoch_name: Optional; name of the epoch to use as a baseline for the AR model.
                               Used only if the data is epoched and concatenate_data is False.
        - concatenate_data: Flag indicating whether to concatenate epoch data if no specific baseline_epoch_name is provided.
                             Used only if the data is epoched.
        - window_size: The size of the window for segmenting the data for local whitening.
                       If None or data is continuous, the entire epoch/data length is used.
        - coeff_mode: Mode for calculating AR coefficients. 'average' uses average coefficients,
                      'separate' calculates coefficients for each channel separately,
                      'first' applies coefficients from the first channel to all channels.

        Note: 
        - If baseline_epoch_name and concatenate_data are not specified, each epoch is whitened based on the AR parameters calculated during the same epoch.
        Further windowing is applied if window_size is set. 

        >>> NOTE: in the average mode make sure that the average is calculated after excluding non-EEG channels
        """

        # Verify whether EEG data is epoched or continuous
        epoched_data_flag = isinstance(self.eeg_data, dict)
        epochs = self.eeg_data if epoched_data_flag else {'continuous': self.eeg_data}

        # Determine the baseline for AR coefficients
        global_coefficients = None
        if baseline_epoch_name and baseline_epoch_name in epochs:
            baseline_data = epochs[baseline_epoch_name]
            global_coefficients = self._calculate_ar_coefficients(baseline_data, AR_order, coeff_mode)
        elif concatenate_data:
            # Concatenate all epochs to create a full baseline dataset
            baseline_data = np.concatenate(list(epochs.values()), axis=0)
            global_coefficients = self._calculate_ar_coefficients(baseline_data, AR_order, coeff_mode)
        
        # Apply AR model coefficients or calculate locally for each window/segment
        whitened_data = {}
        for epoch_name, epoch_data in epochs.items():
            if global_coefficients is not None and window_size is None:
                # Use the global coefficients if they are provided (and applicable) and segmentation is not requested
                whitened_data[epoch_name] = self._apply_whitening_with_global_coefficients(epoch_data, global_coefficients, coeff_mode)                
            else:
                # Calculate local coefficients for each segment. This include the condition when the entire epoch is used for calculating AR parameters and whitening 
                whitened_data[epoch_name] = self._whiten_data_segment(epoch_data, AR_order, window_size, coeff_mode)
        
        # Set the whitened data
        self.whitened_eeg = whitened_data if epoched_data_flag else whitened_data['continuous']
        self.time_domain_whitened = True

    def _apply_whitening_with_global_coefficients(self, epoch_data, coefficients, mode):
        """
        Apply the AR model coefficients to the epoch_data for all channels.
        """
        num_channels = epoch_data.shape[1]
        whitened_epoch_data = np.zeros_like(epoch_data)

        # Apply the global coefficients to each channel of the epoch
        for ch in range(num_channels):
            if mode == 'separate':
                whitened_epoch_data[:, ch] = filtfilt(coefficients[:, ch], [1], epoch_data[:, ch])
            else:
                whitened_epoch_data[:, ch] = filtfilt(coefficients, [1], epoch_data[:, ch])
        
        return whitened_epoch_data

    def _whiten_data_segment(self, data, AR_order, window_size, mode):
        """
        Whiten each channel of the data by calculating AR coefficients and applying the filter segment-wise.
        """
        num_samples, num_channels = data.shape
        whitened_data = np.zeros_like(data)
        window = window_size if window_size is not None else num_samples

        if mode == 'separate':
            for ch in range(num_channels):
                channel_whitened_data = np.zeros(num_samples)
                for seg_start in range(0, num_samples, window):
                    seg_end = min(seg_start + window, num_samples)
                    segment = data[seg_start:seg_end, ch]
                    coefficients = self._calculate_ar_coefficients(segment.reshape(-1, 1), AR_order, mode='separate')
                    channel_whitened_data[seg_start:seg_end] = filtfilt(coefficients.reshape(-1), [1], segment)
                whitened_data[:, ch] = channel_whitened_data
        else:
            for seg_start in range(0, num_samples, window):
                seg_end = min(seg_start + window, num_samples)
                segment = data[seg_start:seg_end]
                coefficients = self._calculate_ar_coefficients(segment, AR_order, mode=mode)
                for ch in range(num_channels):
                    whitened_data[seg_start:seg_end, ch] = filtfilt(coefficients, [1], segment[:, ch])

        return whitened_data

    def _calculate_ar_coefficients(self, data, AR_order, mode='average'):
        """
        Fit an AR model to the provided EEG data and calculate the whitening coefficients.

        Parameters:
        - data: EEG data used as input for the AR model fitting.
        - AR_order: Order of the AR model to be fitted.
        - mode: Specifies how the AR coefficients are calculated across channels.
                'first': Use coefficients from the first channel only.
                'average': Average the coefficients across all channels.
                'separate': Calculate and apply coefficients for each channel separately.

        Returns:
        - Coefficients of the whitening AR model. This will be a 2D array if mode is 'separate',
        otherwise a 1D array.
        """
        num_channels = data.shape[1] if data.ndim > 1 else 1
        
        if mode == 'first':
            model = AutoReg(data[:, 0], lags=AR_order, old_names=False)
            model_fit = model.fit()
            return np.concatenate(([model_fit.params[0]], -model_fit.params[1:]))
        
        # Initialize an array to store coefficients for each channel if separate, or an aggregate array
        coefficients = np.zeros((AR_order + 1, num_channels)) if mode == 'separate' else np.zeros(AR_order + 1)

        # Fit the AR model for each channel separately
        for ch in range(num_channels):
            model = AutoReg(data[:, ch], lags=AR_order, old_names=False)
            model_fit = model.fit()
            channel_coefficients = np.concatenate(([model_fit.params[0]], -model_fit.params[1:]))
            
            if mode == 'separate':
                coefficients[:, ch] = channel_coefficients
            else:
                coefficients += channel_coefficients

        # Average the coefficients across all channels if in 'average' mode
        if mode == 'average':
            coefficients /= num_channels
        
        return coefficients

    def calculate_time_frequency_map(self, method='multitaper', select_channels=None, select_epochs=None):
        """
        Calculate the time-frequency representation (TFR) for the EEG data.

        This method selects the appropriate EEG data (either raw or previously 
        whitened data, if available) and calculates the TFR using the specified method.

        Parameters:
        - method (str): Specifies the method to compute the time-frequency map. Options include
                        'multitaper', 'morlet', and 'spectrogram'. Default is 'multitaper'.
        - select_channels (list): List of channel names or indices to include in the analysis.
                                  If None, all channels are used.
        - select_epochs (list): List of epoch names to include in the analysis. 
                                If None, all epochs are processed.
                                  
        The method updates the self.tfr dictionary with TimeFrequencyRepresentation instances for each epoch.
        If the input EEG data is continuous rather than epoched, the resulting TFR is directly assigned to self.tfr.
        """

        # Determine the data source for TFR calculation
        data_source = self.whitened_eeg if self.whitened_eeg is not None else self.eeg_data

        # Ensure select_channels is a list
        if select_channels is not None and not isinstance(select_channels, (list, tuple)):
            select_channels = [select_channels]

        # Ensure select_epochs is a list
        if select_epochs is not None and not isinstance(select_epochs, (list, tuple)):
            select_epochs = [select_epochs]

        # If data_source is a dictionary 
        if isinstance(data_source, dict):
            epochs_to_process = data_source.keys() if select_epochs is None else select_epochs
            
            # Check if selected_epochs exist in data_source
            if not all(epoch in data_source for epoch in epochs_to_process):
                missing_epochs = [epoch for epoch in epochs_to_process if epoch not in data_source]
                raise ValueError(f'The following selected epochs are not available: {missing_epochs}')

            # Calculate TFR for each epoch in epochs_to_process
            for epoch_idx, epoch_name in enumerate(epochs_to_process):
                epoch_data = data_source[epoch_name]
                self.tfr[epoch_name], chan_names = self._calculate_epoch_tfr(epoch_data, method, select_channels)
                if epoch_idx == 0:
                    self.select_channel_names = chan_names
        # If data_source in continuous EEG data 
        else:
            self.tfr, self.select_channel_names = self._calculate_epoch_tfr(data_source, method, select_channels)

    def _calculate_epoch_tfr(self, epoch_data, method, select_channels):

        ch_names, ch_types, epoch = self._select_channels_and_adjust_data(epoch_data, select_channels)
        raw = create_mne_raw_from_data(
            data=epoch, 
            channel_names=ch_names, 
            sampling_frequency=self.sampling_frequency, 
            ch_types=ch_types
        )

        if method.lower()=='multitaper':
            tfr = self._calculate_multitaper_tfr(raw)
        elif method.lower()=='morlet':
            tfr = self._calculate_morlet_tfr(raw)            
        elif method.lower()=='spectrogram':
            tfr = self._calculate_spectrogram_tfr(raw)
        else:
            raise ValueError("Unsupported time-frequency representation method: {}".format(method))
        
        return tfr, ch_names
    
    def _calculate_multitaper_tfr(self, raw):
        tfr_mne = mne.time_frequency.tfr_multitaper(
            raw, 
            freqs=self.frequencies, 
            n_cycles=self.frequencies*2, 
            time_bandwidth=3, 
            return_itc=False,
            average=False,
            decim=int(self.sampling_frequency * self.time_step),
            verbose=False
        )
        tfr_data = np.transpose(tfr_mne.data, (1, 2, 0))
        tfr_freqs = tfr_mne.freqs
        tfr_times = tfr_mne.times

        return TimeFrequencyRepresentation(tfr_data, tfr_times, tfr_freqs)

    def _calculate_morlet_tfr(self, raw):
        tfr_mne = mne.time_frequency.tfr_morlet(
            raw, 
            freqs=self.frequencies, 
            n_cycles=self.frequencies / 2,
            return_itc=False,
            average=False, 
            decim=int(self.sampling_frequency * self.time_step), 
            n_jobs=1,
            verbose=False
        )
        tfr_data = np.transpose(tfr_mne.data, (1, 2, 0))
        tfr_freqs = tfr_mne.freqs
        tfr_times = tfr_mne.times

        return TimeFrequencyRepresentation(tfr_data, tfr_times, tfr_freqs)
    
    def _calculate_spectrogram_tfr(self, epoch):
        num_time_points, num_chan = epoch.shape
        num_time_steps = int((num_time_points / self.sampling_frequency) / self.time_step)

        # Applying zero-padding, ensuring consistent frequency bin spacing with other analysis methods
        nfft = int(np.ceil(self.sampling_frequency/self.freq_resolution))
        if nfft % 2 != 0:
            nfft += 1

        tfr_data = np.full((len(self.frequencies), num_time_steps, num_chan), np.nan)

        for ch in range(num_chan):
            _, times, Sxx = spectrogram(
                epoch[:, ch],
                fs=self.sampling_frequency,
                window='hann',
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=nfft,
                detrend=False
            )
            freq_indices = np.where(np.isin(_, self.frequencies))[0]
            tfr_data[:, :Sxx.shape[1], ch] = Sxx[freq_indices, :]
        tfr_freqs = self.frequencies
        tfr_times = times  

        return TimeFrequencyRepresentation(tfr_data, tfr_times, tfr_freqs)        

    def _select_channels_and_adjust_data(self, epoch, select_channels):
        return select_channels_and_adjust_data(
            epoch,
            select_channels,
            self.channel_names,
            self.channel_groups
        ) 
        # if select_channels is not None:
        #     if isinstance(select_channels[0], str):
        #         ch_names = select_channels
        #         channel_indices = [self.channel_names.index(name) for name in ch_names]
        #         epoch = epoch[:, channel_indices]
        #     elif isinstance(select_channels[0], int):
        #         ch_names = [self.channel_names[i] for i in select_channels]
        #         epoch = epoch[:, select_channels]
        #     ch_types=['eeg'] * len(select_channels)     
        # else:
        #     ch_names = self.channel_names
        #     ch_types = None

        # return ch_names, ch_types, epoch
    
    def postprocess_anatomical_region_average(self, attr_name=None):
        """
        Calculate an average TFR for pre-defined anatomical channel groups. 
        The average is computed by calling the `calculate_anatomical_group_average` method of the TFR
        objects, which should be instances of the TimeFrequencyRepresentation class.
        
        Parameters:
        - attr_name (str): The attribute name of the TFR data dictionary. Defaults to 'tfr'.
        """
        # Select which TFR dictionary to use based on the use_whitened_tfr parameter
        attr_name = attr_name or 'tfr'

        tfr_dict = getattr(self, attr_name, self.tfr)
        if tfr_dict is None:
            warnings.warn(f"The attribute '{attr_name}' has not been calculated or does not exist. Proceeding with the original TFRs")

        channel_groups = self.channel_groups

        def calculate_region_average(tfr, region_name):
            return tfr.calculate_anatomical_group_average(region_name, self.select_channel_names, channel_groups)

        self.region_average_tfr = {
            region: {
                epoch_name: calculate_region_average(tfr, region)
                for epoch_name, tfr in tfr_dict.items()
            }
            for region in channel_groups
        }

    def postprocess_time_window_average(self, window_size, step_size, attr_name=None):
        """
        Calculate average TFR over time windows.
        """
        tfr_freqs = self.frequencies
        attr_name = attr_name or 'tfr'

        tfr_dict = getattr(self, attr_name, None)
        if tfr_dict is None:
            raise ValueError(f"The attribute {attr_name} has not been calculated or does not exist.")

        is_region_average = next(iter(tfr_dict)) in self.channel_groups

        def calculate_window_average(tfr):
            tfr_mean, _, window_centers = tfr.calculate_window_average(window_size, step_size)
            return TimeFrequencyRepresentation(data=tfr_mean, times=window_centers, frequencies=tfr_freqs)

        if is_region_average:
            self.window_average_tfr = {
                region: {
                    epoch_name: calculate_window_average(tfr)
                    for epoch_name, tfr in region_tfr.items()
                }
                for region, region_tfr in tfr_dict.items()
            }
        else:
            self.window_average_tfr = {
                epoch_name: calculate_window_average(tfr)
                for epoch_name, tfr in tfr_dict.items()
            }

        self.averaging_win_size = window_size
        self.averaging_step_size = step_size

    def postprocess_aperiodic_paramaters(self, freq_range=None, aperiodic_mode='fixed', attr_name=None):
        """
        Calculate FOOOF aperiodic parameters.
        """

        if freq_range is None:
            freq_range = [20, 40]

        attr_name = attr_name or 'tfr'
            
        tfr_dict = getattr(self, attr_name, None)
        if tfr_dict is None:
            raise ValueError(f"The attribute {attr_name} has not been calculated or does not exist")

        is_region_average = next(iter(tfr_dict)) in self.channel_groups

        def calculate_params(tfr):
            return tfr.calculate_aperiodic_parameters(
                freq_range=freq_range, aperiodic_mode=aperiodic_mode
            )

        if is_region_average:
            self.fooof_aperiodic_parameters = {
                region: {
                    epoch_name: calculate_params(tfr)
                    for epoch_name, tfr in region_tfr.items()
                }
                for region, region_tfr in tfr_dict.items()
            }
        else:
            self.fooof_aperiodic_parameters = {
                epoch_name: calculate_params(tfr)
                for epoch_name, tfr in tfr_dict.items()
            }

    def postprocess_periodic_parameters(self, freq_range=None, fbands=None, aperiodic_mode='knee', peak_threshold=1.5, overlap_threshold=0.25, attr_name=None):
        """
        Calculate FOOOF periodic parameters.
        """

        if freq_range is None:
            freq_range = [0.5, self.frequencies[-1]]

        if fbands is None:
            fbands = {
                'delta': [0.5, 4],
                'theta': [4, 7],
                'alpha':[7, 14],
                'beta': [18, 35],
                'gamma':[40, 55]
            }

        attr_name = attr_name or 'tfr'
            
        tfr_dict = getattr(self, attr_name, None)
        if tfr_dict is None:
            raise ValueError(f"The attribute {attr_name} has not been calculated or does not exist")

        is_region_average = next(iter(tfr_dict)) in self.channel_groups

        def calculate_params(tfr):
            return tfr.calculate_periodic_parameters(
                tfr_data = tfr.data,
                freq_range=freq_range, 
                fbands=fbands, 
                aperiodic_mode=aperiodic_mode, 
                peak_threshold=peak_threshold, 
                overlap_threshold=overlap_threshold
            )

        if is_region_average:
            for region, region_tfr in tfr_dict.items():
                region_params = {}
                region_models = {}
                aperiodic_fit = {}
                aperiodic_fit_slope = {}
                for epoch_name, tfr in region_tfr.items():
                    region_params[epoch_name], region_models[epoch_name], aperiodic_fit[epoch_name], aperiodic_fit_slope[epoch_name] = calculate_params(tfr)
                self.fooof_periodic_parameters[region] = region_params
                self.fooof_aperiodic_fit[region] = aperiodic_fit
                self.fooof_aperiodic_fit_slope[region] = aperiodic_fit_slope
                if region_models:  # store if only models are returned
                    self.fooof_model[region] = region_models
        else:
            for epoch_name, tfr in tfr_dict.items():
                self.fooof_periodic_parameters[epoch_name], epoch_fooof_models, self.fooof_aperiodic_fit[epoch_name], self.fooof_aperiodic_fit_slope[epoch_name] = calculate_params(tfr)
                if epoch_fooof_models:
                    self.fooof_model[epoch_name] = epoch_fooof_models

    def postprocess_frequency_domain_whitening(self, baseline_epoch_name=None, use_concatenated_baseline=False, per_channel=False, window_size=300, step_size=60, fbands = None, freq_range=None, fooof_aperidoc_freq_start=None, peak_threshold=0.1, attr_name='tfr'):
        """
        Apply frequency domain whitening on the calculated TFR data.
        
        Parameters: 
        - baseline_epoch_name: Optional; name of the epoch to use as a baseline. If None and
        use_concatenated_baseline is False, each epoch is whitened based on its own data.
        - use_concatenated_baseline: Optional; Boolean flag to indicate whether to use 
        concatenated data from all epochs as a baseline. If baseline_epoch_name is provided,
        this flag is ignored.
        - per_channel: if False the whitening parameters are calculated from the avereage of data from all channels, otherwise calculated for each individual channel
        Regardless, the aperiodic component is applied to individual channels for calculation of whitened TFRs  
        - attr_name: the tfr type (tfr, region_average) that the whitening is applied on
        """
        
        def calculate_baseline_data(tfr_dict, is_region_avg):
            if baseline_epoch_name:
                return {region: tfr_dict[region][baseline_epoch_name].data for region in tfr_dict} if is_region_avg else tfr_dict[baseline_epoch_name].data
            if use_concatenated_baseline:
                if is_region_avg:
                    return {region: np.concatenate([tfr.data for tfr in region_tfr.values()], axis=1) for region, region_tfr in tfr_dict.items()}
                return np.concatenate([tfr.data for tfr in tfr_dict.values()], axis=1)
            return None
        
        def calculate_whitening_parameters(tfr, baseline_data, per_channel, exclude_fbands=True, peak_threshold=0.1, fbands=None, freq_range=None, fooof_aperidoc_freq_start=None, window_size=None, step_size=None, region=None, epoch_name=None):
            
            # TODO: If the periodic parameters have been previously calculated and are suitable for this purpose, avoid recalculating them here. 
            # In such cases, this method should accept the periodic parameters as input.
            
            if baseline_data is not None:
                baseline_tfr = TimeFrequencyRepresentation(
                    data=baseline_data, times=tfr.times, frequencies=tfr.frequencies
                )
                
                baseline_tfr.calculate_whitening_parameters(
                    baseline_data=baseline_data, 
                    per_channel=per_channel, 
                    exclude_fbands=exclude_fbands, 
                    peak_threshold=peak_threshold, 
                    fbands=fbands,
                    freq_range=freq_range,
                    fooof_aperidoc_freq_start=fooof_aperidoc_freq_start
                )
                return baseline_tfr
            
            tfr.calculate_whitening_parameters(
                per_channel=per_channel, 
                exclude_fbands=exclude_fbands, 
                peak_threshold=peak_threshold, 
                window_size=window_size, 
                step_size=step_size, 
                fbands=fbands,
                freq_range=freq_range,
                fooof_aperidoc_freq_start=fooof_aperidoc_freq_start
            )
            return tfr 

        attr_name = attr_name or 'tfr'
        tfr_dict = getattr(self, attr_name, None)

        if fbands is None:
            fbands = {
                'delta': [0.5, 4],
                'theta': [4, 7],
                'alpha':[7, 14],
                'beta': [18, 35],
                'gamma':[35, 55]
            }

        if self.averaging_win_size is None: # if this has not been set somewheere else, then set it using the argument passed to this method
            self.averaging_win_size = window_size

        if self.averaging_step_size is None: # the same as averaging_win_size
            self.averaging_step_size = step_size
        
        if tfr_dict is None:
            raise ValueError(f"The attribute {attr_name} has not been calculated or does not exist")
        
        if self.time_domain_whitened:
            warnings.warn("Time domain whitening has already been applied.", UserWarning)
        
        is_region_avg = next(iter(tfr_dict)) in self.channel_groups
        baseline_data = calculate_baseline_data(tfr_dict, is_region_avg)
        
        if is_region_avg:
            for region, region_tfr in tfr_dict.items():
                
                self.whitened_tfr[region] = {}
                self.regression_ap_fit[region] = {}
                self.regresssion_ap_fit_slope[region] = {}
                self.regression_ap_fit_intercept[region] = {}
                self.regression_ap_fit_included_freqs[region] = {}
                self.fband_power_relative_reg_ap[region] = {}
                self.fooof_periodic_parameters[region] = {}
                self.fooof_model[region] = {}
                self.fooof_aperiodic_fit[region] = {}
                self.fooof_aperiodic_fit_slope[region] = {}

                    
                for epoch_name, tfr in region_tfr.items():
                    tfr_with_whitening_params = calculate_whitening_parameters(
                        tfr, 
                        baseline_data.get(region) if baseline_data is not None else None, 
                        per_channel=per_channel, 
                        exclude_fbands=True, 
                        peak_threshold=peak_threshold, 
                        fbands=fbands,
                        freq_range=freq_range, 
                        fooof_aperidoc_freq_start=fooof_aperidoc_freq_start,
                        window_size=self.averaging_win_size, 
                        step_size=self.averaging_step_size, 
                        # region=region, 
                        # epoch_name=epoch_name,
                    )

                    # Store the aperiodic whitening parameters
                    reg_ap_fit = tfr_with_whitening_params.reg_fit_aperiodic_fit
                    fooof_periodic_params = tfr_with_whitening_params.fooof_periodic_params
                    if baseline_data is None:
                        self.regression_ap_fit[region][epoch_name] = reg_ap_fit
                        self.regresssion_ap_fit_slope[region][epoch_name] = tfr_with_whitening_params.reg_fit_slope
                        self.regression_ap_fit_intercept[region][epoch_name] = tfr_with_whitening_params.reg_fit_intercept
                        self.regression_ap_fit_included_freqs[region][epoch_name] = tfr_with_whitening_params.reg_fit_included_freqs
                        
                        self.fooof_periodic_parameters[region][epoch_name] = fooof_periodic_params
                        self.fooof_model[region][epoch_name] = tfr_with_whitening_params.fooof_model
                        self.fooof_aperiodic_fit[region][epoch_name] = tfr_with_whitening_params.fooof_aperiodic_fit
                        self.fooof_aperiodic_fit_slope[region][epoch_name] = tfr_with_whitening_params.fooof_aperiodic_fit_slope
                    
                    else: # if baseline is used
                        pass
                        #TODO: window the data based on the win_size and win_step_size that are passed to the method
                        #TODO: calculate FOOOF periodic parameters for no baseline condition. 

                    # Use the calculated aperiodic component to whiten the spectrum and calculate relative power in frequency bands detected using FOOOF
                    whitened_tfr = tfr.calculate_whitened_tfr(
                        reg_ap_fit, self.averaging_win_size, self.averaging_step_size, per_channel
                    )
                    self.whitened_tfr[region][epoch_name] = whitened_tfr


                    # Calculate the relative power for each frequency band. The boudnaries of each frequency band is calcualted using FOOOF
                    # NOTE: the boundaries of each frequency band might have been extracted from the baseline or the current epoch, and used for the calculation of the power in corresponding band. 
                    window_average_whitened_tfr_data, _, _ = whitened_tfr.calculate_window_average(
                        window_size=self.averaging_win_size, step_size= self.averaging_step_size
                    )
                    
                    tfr_freqs = whitened_tfr.frequencies

                    self.fband_power_relative_reg_ap[region][epoch_name] = {}
                    num_channels, num_time_points = fooof_periodic_params['alpha']['lower_bound'].shape
                    for fband_label, fband_bound in fbands.items():
                        self.fband_power_relative_reg_ap[region][epoch_name][fband_label] = np.full((num_channels, num_time_points), np.nan)
                        for chan_idx in range(num_channels):
                            for time_pt in range(num_time_points):
                                if fband_label != 'delta':
                                        lower_bound = fooof_periodic_params[fband_label]['lower_bound'][chan_idx, time_pt]
                                        upper_bound = fooof_periodic_params[fband_label]['upper_bound'][chan_idx, time_pt]
                                else:
                                    lower_bound, upper_bound = fband_bound

                                if lower_bound is not None and upper_bound is not None:
                                    band_mask = (tfr_freqs >= lower_bound) & (tfr_freqs <= upper_bound)
                                    power_relative = np.mean(window_average_whitened_tfr_data[band_mask, time_pt, chan_idx]) if band_mask.any() else np.nan
                                    self.fband_power_relative_reg_ap[region][epoch_name][fband_label][chan_idx, time_pt] = power_relative

        else:
            for epoch_name, tfr in tfr_dict.items():
                tfr_with_whitening_params = calculate_whitening_parameters(
                    tfr, 
                    baseline_data, 
                    per_channel=per_channel,
                    exclude_fbands=True,
                    peak_threshold=peak_threshold,
                    fbands=fbands,
                    freq_range=freq_range,
                    fooof_aperidoc_freq_start = fooof_aperidoc_freq_start,
                    window_size=self.averaging_win_size,
                    step_size=self.averaging_step_size,
                    # region=region, 
                    # epoch_name=epoch_name,
                )

                # Store the aperiodic whitening parameters
                reg_ap_fit = tfr_with_whitening_params.reg_fit_aperiodic_fit
                if baseline_data is None:
                    self.regression_ap_fit[epoch_name] = reg_ap_fit
                    self.regresssion_ap_fit_slope[epoch_name] = tfr_with_whitening_params.reg_fit_slope
                    self.regression_ap_fit_intercept[epoch_name] = tfr_with_whitening_params.reg_fit_intercept
                    self.regression_ap_fit_included_freqs[epoch_name] = tfr_with_whitening_params.reg_fit_included_freqs
                    
                    self.fooof_periodic_parameters[epoch_name] = tfr_with_whitening_params.fooof_periodic_params
                    self.fooof_model[epoch_name] = tfr_with_whitening_params.fooof_model
                    self.fooof_aperiodic_fit[epoch_name] = tfr_with_whitening_params.fooof_aperiodic_fit
                    self.fooof_aperiodic_fit_slope[epoch_name] = tfr_with_whitening_params.fooof_aperiodic_fit_slope

                else: # if baseline is used
                    pass
                    #TODO: window the data based on the win_size and win_step_size that are passed to the method
                    #TODO: calculate FOOOF periodic parameters for no baseline condition. 

                # Use the calculated aperiodic component to whiten the spectrum and calculate relative power in specific frequency bands
                whitened_tfr = tfr.calculate_whitened_tfr(
                    reg_ap_fit, self.averaging_win_size, self.averaging_step_size, per_channel
                )
                self.whitened_tfr[epoch_name] = whitened_tfr

                window_average_whitened_tfr_data, _, _ = whitened_tfr.calculate_window_average(
                    window_size=self.averaging_win_size, step_size= self.averaging_step_size
                )
                    
                tfr_freqs = whitened_tfr.frequencies

                self.fband_power_relative_reg_ap[epoch_name] = {}
                num_channels, num_time_points = fooof_periodic_params['alpha']['lower_bound'].shape
                for fband_label, fband_bound in fbands.items():
                    self.fband_power_relative_reg_ap[epoch_name][fband_label] = np.full((num_channels, num_time_points), np.nan)
                    for chan_idx in range(num_channels):
                        for time_pt in range(num_time_points):
                            if fband_label != 'delta':
                                    lower_bound = fooof_periodic_params[fband_label]['lower_bound'][chan_idx, time_pt]
                                    upper_bound = fooof_periodic_params[fband_label]['upper_bound'][chan_idx, time_pt]
                            else:
                                lower_bound, upper_bound = fband_bound

                            if lower_bound is not None and upper_bound is not None:
                                band_mask = (tfr_freqs >= lower_bound) & (tfr_freqs <= upper_bound)
                                power_relative = np.mean(window_average_whitened_tfr_data[band_mask, time_pt, chan_idx]) if band_mask.any() else np.nan
                                self.fband_power_relative_reg_ap[epoch_name][fband_label][chan_idx, time_pt] = power_relative

                
    def calculate_emergence_trajectory(self):
        """
        Calculate power for each frequency band.
        """
        # Placeholder for actual spectral parameter calculation logic
        self.emergence_trajectory = {}  # Replace with actual spectral parameter calculation




import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import hamming

class MetastableSpectralDynamicAnalysis:
    def __init__(self, participant_id, data):
        self.participant_id = participant_id
        self.data = data
        self.spectral_power_vectors = None
        self.cleaned_vectors = None
        self.pca_vectors = None
        self.cluster_labels = None
        self.cluster_quality = None
        self.cluster_consistency = None

    def construct_spectral_power_vectors(self, window_size, frequency_intervals):
        """
        Construct spectral power vectors for each time window.

        Parameters:
        window_size (int): Size of each time window.
        frequency_intervals (list): List of tuples defining frequency intervals.
        """
        # Placeholder for constructing spectral power vectors
        num_windows = self.data.shape[0]
        num_channels = self.data.shape[1]
        self.spectral_power_vectors = np.zeros((num_windows, len(frequency_intervals) * num_channels))
        # Loop over each time window
        for i in range(num_windows):
            # Calculate spectral power for each frequency interval and channel
            for j, (low, high) in enumerate(frequency_intervals):
                # Placeholder: Calculate spectral power for the window
                spectral_power = np.random.rand(num_channels)  # Replace with actual calculation
                self.spectral_power_vectors[i, j * num_channels:(j + 1) * num_channels] = spectral_power

    def remove_outlier_windows(self):
        """
        Remove outlier windows based on some criterion.
        """
        # Placeholder for outlier removal logic
        self.cleaned_vectors = self.spectral_power_vectors  # Placeholder: Replace with actual outlier removal

    def apply_pca(self, num_components):
        """
        Apply PCA for dimensionality reduction.

        Parameters:
        num_components (int): Number of principal components to retain.
        """
        pca = PCA(n_components=num_components)
        self.pca_vectors = pca.fit_transform(self.cleaned_vectors)

    def categorize_clusters(self, num_clusters):
        """
        Categorize dimensionality-reduced feature vectors into clusters.

        Parameters:
        num_clusters (int): Number of clusters.
        """
        kmeans = KMeans(n_clusters=num_clusters)
        self.cluster_labels = kmeans.fit_predict(self.pca_vectors)
        # Calculate silhouette score for cluster quality
        self.cluster_quality = silhouette_score(self.pca_vectors, self.cluster_labels)

    def determine_cluster_consistency(self, other_objects):
        """
        Determine cluster consistency between participants.

        Parameters:
        other_objects (list): List of other MetastableSpectralDynamicAnalysis objects.
        """
        # Placeholder for cluster consistency determination logic
        pass  # Replace with actual consistency determination