import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy.signal import filtfilt, spectrogram
from src.eeg_analysis.utils.helpers import create_mne_raw_from_data 
import warnings
from statsmodels.tsa.ar_model import AutoReg
from fooof import FOOOF, FOOOFGroup
from fooof.sim.gen import gen_aperiodic

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
        _slope (float): Parameter for the frequency-domain whitening, representing the slope of the 1/f curve.
        _intercept (float): Parameter for the frequency-domain whitening, representing the intercept of the 1/f curve.

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
        self._slope = None
        self._intercept = None

    def plot(self, ax=None, channel=None, start_time=0, vmin=None, vmax=None, cmap='viridis', colorbar=False, add_lables=False, title='Time-Frequency Representation'):
        
        if channel is not None:
            data_to_plot = self.data[:,:, channel]
        else:
            # Check if the data has more than one channel
            if self.data.ndim == 3 and self.data.shape[2] > 1:
                raise ValueError("Multiple channels detected. Please select a channel to plot.")
            # Otherwise, use the data as is (single channel case)
            data_to_plot = self.data
        
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

    def calculate_whitened_tfr(self, baseline_data=None):
        """
        Create a whitened compy of the TFR data and returns it.
        """
        tfr_whitened = TimeFrequencyRepresentation(np.copy(self.data), self.times, self.frequencies)
        tfr_whitened.whiten(baseline_data)

        return tfr_whitened

    def whiten(self, baseline_data=None):
        """
        Apply whitening to the TFR data using a baseline or the entire dataset
        to calculate the whitening parameters.

        - baseline_data: Optional; a 3D numpy array of TFR data from a baseline epoch, 
                         or a concatenation of multiple epochs if provided.
        This method alters the TFR data in place.

        It sets `self.whitened_data` with the whitened TFR data.
        """
        self._calculate_whitening_parameters(baseline_data)
        
        # log_freqs = np.log10(self.frequencies)[:, np.newaxis, np.newaxis]  # Reshape for broadcasting
        freqs = self.frequencies[:, np.newaxis]
        tfr_data_db = 10 * np.log10(np.abs(self.data))
        tfr_db_whitened = tfr_data_db - (self._slope * freqs + self._intercept)
        self.data = 10 ** (tfr_db_whitened / 10)

    def _calculate_whitening_parameters(self, baseline_data=None):
        """
        Calculates whitening parameters based on the provided baseline data or the current instance data.
        """
        if baseline_data is None:
            baseline_data = self.data

        freqs = self.frequencies

        # Identify non-outlier time points and calculate mean spectral power
        mean_spectral_power, _ = TimeFrequencyRepresentation._calculate_mean_spectral_power_with_outlier_removal(
            baseline_data
        )
        tfr_db = 10 * np.log10(mean_spectral_power)

        # Fit a linear model (in log-log space) to calculate slope and intercept
        # log_freqs = np.log10(freqs)
        # slope, intercept = np.polyfit(log_freqs, tfr_db, 1)
        slope, intercept = np.polyfit(freqs, tfr_db, 1)

        self._slope = slope
        self._intercept = intercept
    
    def calculate_window_average(self, window_size, step_size):
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
                tfr_mean[:, win_idx, chn_idx] = mean_power
                tfr_std[:, win_idx, chn_idx] = std_power

        return tfr_mean, tfr_std, np.array(window_centers)

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
            fooof_model.fit(tfr_freqs, tfr_data[:, :, chn_idx].T, freq_range=freq_range)

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
    
    def calculate_periodic_parameters(self, freq_range=None, bands=None, aperiodic_mode='knee', peak_threshold=1.5, overlap_threshold=0.25):
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
        
        if not bands:
            raise ValueError("Bands need to be defiend as a dictionary with band names and frequency ranges.")

        tfr_data = self.data
        if tfr_data.ndim == 2: 
            tfr_data = tfr_data[..., np.newaxis]
        
        _, num_time_points, num_channels = tfr_data.shape
        tfr_freqs = self.frequencies

        # Initialize a structure to store FOOOF models only if the conditions are met
        if num_time_points <= 50 and num_channels == 1:
            fooof_models = {}
        else:
            warnings.warn(f"Not storing the fooof_models due to the number of spectrums given the number of time windows ({num_time_points}) and channels ({num_channels}).")
            fooof_models = None
        
         # Initialize a structure to store the periodic parameters for all channels, time points, and bands.
        nan_array = np.full((num_channels, num_time_points), np.nan)
        fooof_periodic_params = {
            band: {
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
            for band in bands
        }

        for chn_idx in range(num_channels):
            for time_pt in range(num_time_points):
                # Fit FOOOF model
                fooof_model = FOOOF(aperiodic_mode=aperiodic_mode, peak_threshold=peak_threshold, verbose=False)
                fooof_model.fit(tfr_freqs, tfr_data[:, time_pt, chn_idx], freq_range=freq_range)

                if fooof_models is not None:
                    fooof_models[(chn_idx, time_pt)] = fooof_model
                
                # aperiodic_params = fooof_model.get_params('aperiodic_params')
                # aperiodic_fit = self._aperiodic_fit(tfr_freqs, aperiodic_params)
                # aperiodic_fit = fooof_model._ap_fit
                aperiodic_fit = gen_aperiodic(fooof_model.freqs, fooof_model._robust_ap_fit(fooof_model.freqs, fooof_model.power_spectrum))

                # Loop over bands and store the peak parameters
                for band_label, freq_band in bands.items():
                    if band_label != 'delta': 
                        extracted_peak = self._extract_highest_peak_in_band(fooof_model, freq_band, overlap_threshold=overlap_threshold)
                        if extracted_peak:         
                            fooof_periodic_params[band_label]['cf'][chn_idx, time_pt] = extracted_peak[0][0]
                            fooof_periodic_params[band_label]['amplitude'][chn_idx, time_pt] = extracted_peak[0][1]
                            fooof_periodic_params[band_label]['bw'][chn_idx, time_pt] = extracted_peak[0][2]
                            fooof_periodic_params[band_label]['lower_bound'][chn_idx, time_pt] = extracted_peak[0][3]
                            fooof_periodic_params[band_label]['upper_bound'][chn_idx, time_pt] = extracted_peak[0][4]

                            lower_bound, upper_bound = extracted_peak[0][3], extracted_peak[0][4]
                        else:
                            lower_bound, upper_bound = freq_band[0], freq_band[1]
                    else:
                        lower_bound, upper_bound = freq_band[0], freq_band[1]

                    band_mask = (tfr_freqs >= lower_bound) & (tfr_freqs <= upper_bound)

                    # Calculate absolute and relative (to the aperiodic component) power
                    if band_mask.any():
                        power_absolute = np.mean(fooof_model.power_spectrum[band_mask])
                        power_relative = np.mean(fooof_model.power_spectrum[band_mask] - aperiodic_fit[band_mask])
                    else:
                        power_absolute = np.nan
                        power_relative = np.nan
                    
                    fooof_periodic_params[band_label]['avg_power_absolute'][chn_idx, time_pt] = power_absolute
                    fooof_periodic_params[band_label]['avg_power_relative'][chn_idx, time_pt] = power_relative
        
        return fooof_periodic_params, fooof_models

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
    
    def _extract_highest_peak_in_band(self, fooof_model, freq_band, overlap_threshold):
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
                X = 1/4
                full_bw = 2 * gaussian_kernel_sd * np.sqrt(-2 * np.log(X))
                # full_bw = 4*gaussian_kernel_sd

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
    def _calculate_mean_spectral_power_with_outlier_removal(tfr_data):
        """
        Identifies outlier time points and calculates the mean spectral power,
        excluding those outliers.
        """
        if tfr_data.ndim < 3: #if processing channel average data
            tfr_data = tfr_data[..., np.newaxis]

        tfr_magnitude = np.abs(tfr_data)
        sum_over_freq_and_channels = np.sum(tfr_magnitude, axis=(0, 2))

        non_outlier_time_indices = TimeFrequencyRepresentation._identify_non_outlier_time_indices(sum_over_freq_and_channels)
        tfr_filtered = tfr_magnitude[:, non_outlier_time_indices, :]

        mean_spectral_power = np.mean(tfr_filtered, axis=(1, 2))
        std_spectral_power = np.std(tfr_filtered, axis=(1, 2))

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

    def __init__(self, eeg_file, window_size=None, step_size=None):
        """
        Initialize the PowerSpectralAnalysis with (preprocessed) EEG data.
        """
        self.participant_id = eeg_file.participant_id
        self.eeg_data = eeg_file.eeg_data
        self.channel_names = eeg_file.channel_names
        self.channel_groups = eeg_file.channel_groups
        self.whitened_eeg = None
        self.time_domain_whitened = False
        self.sampling_frequency = eeg_file.ds_sampling_frequency if hasattr(eeg_file, 'ds_sampling_frequency') else eeg_file.sampling_frequency      
        self.freq_resolution = 0.5
        self.frequencies = np.arange(0.5, 55.5, self.freq_resolution) 

        window_size_samples = window_size*self.sampling_frequency if window_size else int(self.sampling_frequency)
        step_size_samples = step_size*self.sampling_frequency if step_size else window_size_samples // 2
        self.nperseg = window_size_samples
        self.noverlap = step_size_samples

        self.time_step = (self.nperseg - self.noverlap) / self.sampling_frequency
        self.tfr = {}
        self.whitened_tfr = {}
        
        self.region_average_tfr = {}
        self.window_average_tfr = {}

        self.fooof_aperiodic_parameters = {}
        self.fooof_periodic_parameters = {}
        self.fooof_models = None

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

        >>> TO DO: in the average mode make sure that the average is calculated after excluding non-EEG channels
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
            for epoch_name in epochs_to_process:
                epoch_data = data_source[epoch_name]
                self.tfr[epoch_name], self.channel_names = self._calculate_epoch_tfr(epoch_data, method, select_channels)
        
        # If data_source in continuous EEG data 
        else:
            self.tfr, self.channel_names = self._calculate_epoch_tfr(data_source, method, select_channels)

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
            tfr = self._calculate_spectrogram_tfr(epoch)
        else:
            raise ValueError("Unsupported time-frequency representation method: {}".format(method))
        
        return tfr, ch_names
    
    def _calculate_multitaper_tfr(self, raw):
        tfr_mne = mne.time_frequency.tfr_multitaper(
            raw, 
            freqs=self.frequencies, 
            n_cycles=self.frequencies / 2, 
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
        
        if select_channels is not None:
            if isinstance(select_channels[0], str):
                ch_names = select_channels
                channel_indices = [self.channel_names.index(name) for name in ch_names]
                epoch = epoch[:, channel_indices]
            elif isinstance(select_channels[0], int):
                ch_names = [self.channel_names[i] for i in select_channels]
                epoch = epoch[:, select_channels]
            ch_types=['eeg'] * len(select_channels)     
        else:
            ch_names = self.channel_names
            ch_types = None

        return ch_names, ch_types, epoch
    
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
            return tfr.calculate_anatomical_group_average(region_name, self.channel_names, channel_groups)

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

    def postprocess_periodic_parameters(self, freq_range=None, bands=None, aperiodic_mode='knee', peak_threshold=1.5, overlap_threshold=0.25, attr_name=None):
        """
        Calculate FOOOF periodic parameters.
        """

        if freq_range is None:
            freq_range = [0.5, self.frequencies[-1]]

        if bands is None:
            bands = {
                'delta': [0.5, 4],
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
            periodic_params, fooof_models = tfr.calculate_periodic_parameters(
                freq_range=freq_range, bands=bands, aperiodic_mode=aperiodic_mode, peak_threshold=peak_threshold, overlap_threshold=overlap_threshold
            )
            return periodic_params, fooof_models

        fooof_periodic_parameters = {}
        fooof_models = {}

        if is_region_average:
            for region, region_tfr in tfr_dict.items():
                region_params = {}
                region_models = {}
                for epoch_name, tfr in region_tfr.items():
                    region_params[epoch_name], region_models[epoch_name] = calculate_params(tfr)
                fooof_periodic_parameters[region] = region_params
                if region_models: # store if only models are returned
                    fooof_models[region] = region_models
        else:
            for epoch_name, tfr in tfr_dict.itmes():
                fooof_periodic_parameters[epoch_name], fooof_models = calculate_params(tfr)
                if fooof_models:
                    fooof_models[epoch_name] = fooof_models
        
        self.fooof_periodic_parameters = fooof_periodic_parameters
        self.fooof_models = fooof_models

    def postprocess_frequency_domain_whitening(self, baseline_epoch_name=None, use_concatenated_baseline=False, attr_name=None):
        """
        Apply frequency domain whitening on the calculated TFR data.
        This method calcualtes the whitened TFRs by calling the 'calculate_whitened_tfr' method of the TFR objects,
        which should be instances of the TimeFrequencyRepresentation class. 

        Parameters: 
        - baseline_epoch_name: Optional; name of the epoch to use as a baseline. If None, and
          use_concatenated_baseline is False, each epoch is whitened based on its own data.
        - use_concatenated_baseline: Optional; Boolean flag to indicate whether to use 
          concatenated data from all epochs as a baseline. If baseline_epoch_name is provided,
          this flag is ignored.
        """
        attr_name = attr_name or 'tfr'

        tfr_dict = getattr(self, attr_name, None)
        if tfr_dict is None:
            raise ValueError(f"The attribute {attr_name} has not been calculated or does not exist")

        is_region_average = next(iter(tfr_dict)) in self.channel_groups

        if self.time_domain_whitened:
            warnings.warn("Time domain whitening has already been applied.", UserWarning)

        baseline_data = None
        if baseline_epoch_name:
            if is_region_average:
                baseline_data = {region: tfr_dict[region][baseline_epoch_name].data for region in tfr_dict}
            else:
                baseline_data = tfr_dict[baseline_epoch_name].data
        elif use_concatenated_baseline:
            # Concatenate data from all epochs to create a baseline for whitening
            if is_region_average:
                concatenated_data = {
                    region: np.concatenate([tfr.data for tfr in region_tfr.values()], axis=1) for region, region_tfr in tfr_dict.items()
                }
                baseline_data = concatenated_data
            else:
                concatenated_data = [tfr.data for tfr in tfr_dict.values()]
                baseline_data = np.concatenate(concatenated_data, axis=1)

        if is_region_average:
            self.whitened_tfr = {
                region: {
                    epoch_name: tfr.calculate_whitened_tfr(baseline_data[region] if baseline_data else None)
                    for epoch_name, tfr in region_tfr.items()
                }
                for region, region_tfr in tfr_dict.items()
            }
        else:
            self.whitened_tfr = {
                epoch_name: tfr.calculate_whitened_tfr(baseline_data)
                for epoch_name, tfr in tfr_dict.items()
            }

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
