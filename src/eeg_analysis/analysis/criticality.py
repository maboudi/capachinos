import numpy as np
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.stats import pearsonr
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from numpy.typing import NDArray
import powerlaw
from powerlaw import trim_to_range
import statsmodels.api as sm
from warnings import warn
from collections import Counter
from src.eeg_analysis.preprocessing.eeg_preprocessor import EEGPreprocessor
from src.eeg_analysis.utils.helpers import get_eeg_channel_indices, remove_outliers, calculate_z_score_eeg
from collections import defaultdict
from scipy.signal import correlate, correlation_lags
import itertools

# import powerlaw
import matplotlib.pyplot as plt

class EEGSignal:
    def __init__(self, signal, sampling_rate):
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.duration = len(signal) / sampling_rate

    def bandpass_filter(self, lowcut, highcut, order=3):
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, self.signal, padlen=150)
        return filtered_signal

    def hilbert_transform(self, signal):
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        return envelope
    
    def mean_and_variance(self, x_t):
        mu = np.mean(x_t)
        nu = np.var(x_t)
        return mu, nu

    def autocorrelation_function(self, x_t, N, mu, nu):
        ACF_values = []
        for s in range(1, N//2):
            ACF_num = np.sum((x_t[:-s] - mu) * (x_t[s:] - mu))
            ACF_value = ACF_num / nu
            ACF_values.append(ACF_value)
        return ACF_values


class AutoCorrelationFunction:
    # Placeholder for Autocorrelation function 

    def __init__(self, eeg_signal, frequency_bands):
        self.eeg_signal = eeg_signal
        self.frequency_bands = frequency_bands

    def perform_acf_analysis(self):
        # Bandpass filter the EEG in each frequency band of interest
        results = {}
        for name, (lowcut, highcut) in self.frequency_bands.items():
            filtered_signal = self.eeg_signal.bandpass_filter(lowcut, highcut)
            envelope = self.eeg_signal.hilbert_transform(filtered_signal)
            
            mu, nu = self.eeg_signal.mean_and_variance(envelope)
            ACF_values = self.eeg_signal.autocorrelation_function(envelope, len(envelope), mu, nu)
            
            results[name] = ACF_values
        
        return results    

    def plot_acf_heatmap(self, acf_values):
        # Create a time-frequency heatmap of ACF(1) values
        freq_band_names = list(acf_values.keys())
        ACF_at_lag_one = [acf[0] for acf in acf_values.values()]
        
        plt.figure(figsize=(10, 5))
        plt.imshow([ACF_at_lag_one], aspect='auto', cmap='hot', extent=(0, self.eeg_signal.duration, 0, len(freq_band_names)))
        plt.yticks(ticks=np.arange(len(freq_band_names)), labels=freq_band_names)
        plt.colorbar(label='ACF(1)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency Band')
        plt.title('Time-Frequency Heatmap of ACF(1)')
        plt.show()


class PhaseLagEntropy:
    def __init__(self, eeg_signal):
        self.eeg_signal = eeg_signal

    def phase_lag_entropy(self):
        # Criticality analysis - Phase lag entropy topography
        raise NotImplementedError()

    def instantaneous_phase(self, signal):
        analytic_signal = hilbert(signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        return instantaneous_phase

    def calculate_entropy(self, phase_difference_patterns):
        pattern_probs = self.calculate_pattern_probability(phase_difference_patterns)
        entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in pattern_probs])
        normalized_entropy = entropy / np.log2(len(phase_difference_patterns))
        return normalized_entropy
    
    # Adding additional methods for phase lag entropy
    def calculate_pattern_probability(self, patterns):
        unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
        probabilities = counts / counts.sum()
        return dict(zip(map(tuple, unique_patterns), probabilities))

    def phase_lag_entropy_topography(self, eeg_signals):
        # Given a set of EEG signals as channels, calculate phase lag entropy matrix
        num_channels = len(eeg_signals)
        entropy_matrix = np.zeros((num_channels, num_channels))
        for i in range(num_channels):
            for j in range(num_channels):
                if i != j:
                    phase_i = self.instantaneous_phase(eeg_signals[i])
                    phase_j = self.instantaneous_phase(eeg_signals[j])
                    phase_diff = phase_i - phase_j
                    
                    # Binarization of the phase difference
                    patterns = [binarize_phase(diff) for diff in phase_diff]
                    entropy = self.calculate_entropy(patterns)
                    entropy_matrix[i, j] = entropy
        
        entropy_topography = np.mean(entropy_matrix, axis=1)
        return entropy_topography

@dataclass
class PowerLawFitResult:
    """
    This class stores the results of power-law fit.
    """
    alpha: float
    D: float
    sigma: float
    xmin: float
    xmax: float
    R_exp: float
    p_exp: float
    p: float

    alphas: List[float] = field(default_factory=list)
    Ds: List[float] = field(default_factory=list)
    sigmas: List[float] = field(default_factory=list)
    xmins: List[float] = field(default_factory=list)
    xmaxs: List[float] = field(default_factory=list)
    Rs_exp: List[float] = field(default_factory=list)
    ps_exp: List[float] = field(default_factory=list)
    ps: List[float] = field(default_factory=list)

    alpha_2d: NDArray[np.float64] = field(default_factory=lambda: np.full((0, 0), np.nan))
    D_2d: NDArray[np.float64] = field(default_factory=lambda: np.full((0, 0), np.nan))
    sigma_2d: NDArray[np.float64] = field(default_factory=lambda: np.full((0, 0), np.nan))
    R_exp_2d: NDArray[np.float64] = field(default_factory=lambda: np.full((0, 0), np.nan))


class PowerLawAnalyzer:
    def __init__(self, data: np.ndarray):
        self.data = data

    def generate_log_space_values(self, start_value: float, stop_value:float, num: int):
        if start_value == 0:
            warn('The start value should not be zero due to logarithmic operations. The method returns NaNs.')
        # return np.logspace(np.log10(start_value), np.log10(stop_value), num)
        return np.linspace(start_value, stop_value, num)

    def generate_and_sort_fit_ranges(self, values: NDArray) -> List[Tuple[float, float]]:
        """ 
        generate the ranges of parameter values
        over which the power law fit exponent will be calculated. 
        Its highly similar to the definition of the supports 
        (pairs of values defining the fit range) are defined in NCC toolbox. 
        """
        pairs = list(itertools.product(values, repeat=2))  # Generate all combinations of values with themselves

        # allocate differnet set of values to xmin and xmax
        # mid_index = len(values) // 2
        # first_half = values[:mid_index]
        # second_half = values[mid_index:]
        # pairs = list(itertools.product(first_half, second_half)) # Generate all combinations of first_half values with second_half values

        # Constrain to fit within region data
        constrained_pairs = [
            pair for pair in pairs 
            if pair[0] >= np.min(self.data) 
            and pair[1] <= np.max(self.data) 
            and pair[1] > pair[0]
        ]
        # if not constrained_pairs:
        #     constrained_pairs = [[max(first_half), min(second_half)]]

        try:
            sorted_pairs = sorted(constrained_pairs, key=lambda pair: (np.log(pair[1]) - np.log(pair[0])) ** 2, reverse=True)
        except:
            sorted_pairs = constrained_pairs
        
        return sorted_pairs

    def compute_power_law_fits(self, start_value: float = None, stop_value: float = None, num_values: int = 20) -> PowerLawFitResult:
        
        if start_value is None:
            start_value = np.min(self.data)
        if stop_value is None:
            stop_value = np.max(self.data)
        
        if len(set(self.data)) < 50:
            values = np.array(list(set(self.data)))
        else:
            values = self.generate_log_space_values(start_value, stop_value, num_values)
        
        fit_ranges = self.generate_and_sort_fit_ranges(values)

        alphas, Ds, sigmas, Rs_exp, ps_exp, ps, xmins, xmaxs = [], [], [], [], [], [], [], []

        unique_xmins = sorted(set(x for x, _ in fit_ranges))
        unique_xmaxs = sorted(set(x for _, x in fit_ranges))

        xmin_to_index = {xmin: i for i, xmin in enumerate(unique_xmins)}
        xmax_to_index = {xmax: j for j, xmax in enumerate(unique_xmaxs)}

        alpha_2d = np.full((len(unique_xmins), len(unique_xmaxs)), np.nan)
        D_2d = np.full((len(unique_xmins), len(unique_xmaxs)), np.nan)
        sigma_2d = np.full((len(unique_xmins), len(unique_xmaxs)), np.nan)
        R_exp_2d = np.full((len(unique_xmins), len(unique_xmaxs)), np.nan)

        for xmin, xmax in fit_ranges:
            
            # check if enough data is remained to fit power law, after trimming the data to the range (xmin, xmax)
            data_trimmed = trim_to_range(self.data, xmin, xmax)
            if (len(data_trimmed) < (len(self.data) / 2)) or (len(np.unique(data_trimmed)) < 3):
                continue

            fit = powerlaw.Fit(data_trimmed, xmin=xmin, xmax=xmax, discrete=True, verbose=False)

            xmins.append(xmin)
            xmaxs.append(xmax)

            alphas.append(fit.power_law.alpha)
            Ds.append(fit.power_law.D)
            sigmas.append(fit.power_law.sigma)

            R, p_exp = fit.distribution_compare('power_law', 'exponential')
            Rs_exp.append(R)
            ps_exp.append(p_exp)

            i = xmin_to_index[xmin]
            j = xmax_to_index[xmax]

            alpha_2d[i, j] = fit.power_law.alpha
            D_2d[i, j] = fit.power_law.D
            sigma_2d[i, j] = fit.power_law.sigma
            R_exp_2d[i, j] = R

            # We might calcualte and append the p-value for the power-law fit
            # p = self._calculate_powerlaw_fit_p_value(fit, data, xmin, xmax)
            # ps.append(p)

        D = min(Ds)
        min_D_idx = Ds.index(D)
        alpha = alphas[min_D_idx]
        sigma = sigmas[min_D_idx]
        R_exp = Rs_exp[min_D_idx]
        p_exp = ps_exp[min_D_idx]
        xmin = xmins[min_D_idx]
        xmax = xmaxs[min_D_idx]
        # p = ps[min_D_idx]

        return PowerLawFitResult(
            alpha=alpha,
            D=D,
            sigma=sigma,
            R_exp=R_exp,
            p_exp=p_exp,
            p=None, # Calculate if needed
            xmin=xmin,
            xmax=xmax,
            alphas=alphas,
            Ds=Ds,
            sigmas=sigmas,
            Rs_exp=Rs_exp,
            ps_exp=ps_exp,
            ps=ps, # Populate if p-values are to be included
            xmins=xmins,
            xmaxs=xmaxs,
            alpha_2d=alpha_2d,
            D_2d=D_2d,
            sigma_2d=sigma_2d,
            R_exp_2d=R_exp_2d
        )
    
    def _calculate_powerlaw_fit_p_value(self, fit, data, xmin, xmax, num_bootstraps=100):
        """
        Calculate the p-value of the power-law fit using bootstrapping.
        
        Parameters:
        fit: powerlaw.Fit object
        num_bootstraps: int, number of bootstrap samples to use (default 1000)
        
        Returns:
        p: float, p-value of the fit
        """
        # Get the KS statistic for the power-law fit
        ks_stat = fit.power_law.D
        
        # Generate bootstrap samples
        bootstrap_ks_stats = []
        for _ in range(num_bootstraps):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_fit = powerlaw.Fit(bootstrap_sample, xmin=xmin, xmax=xmax, discrete=True, verbose=False)
            bootstrap_ks_stat = bootstrap_fit.power_law.D
            bootstrap_ks_stats.append(bootstrap_ks_stat)
        
        # Calculate the p-value
        p_value = np.sum(np.array(bootstrap_ks_stats) > ks_stat) / num_bootstraps
        return p_value
    

class AvalancheDetectionParams:
    def __init__(self, 
                 threshold: float = 2.0, 
                 method: str ='beggs', 
                 epoch_names: Optional[List[str]] = None,
                 inter_event_interval: int = 8,
                 bin_size: int = 2,
                 use_pearson_correlation: bool = True, 
                 correlation_threshold: float = 0.75, 
                 event_boundary: str ='zero_crossing'):
        self.threshold = threshold
        self.method = method
        self.epoch_names = epoch_names
        self.inter_event_interval = inter_event_interval
        self.bin_size = bin_size
        self.use_pearson_correlation = use_pearson_correlation
        self.correlation_threshold = correlation_threshold
        self.event_boundary = event_boundary


@dataclass
class AvalancheParams:
    """
    This class stores the parameters related to avalanches. including durations, sizes, 
    and power-law fitting results.
    """
    duration: List[float]
    sum_raw: List[float]
    sum_z: List[float]
    event_count: List[int]
    min_silence: List[float]
    chan_count: List[float]
    region_count: List[int]  
    dominant_region: list[str]

    occurrence_frequency: float
    avg_duration: float
    avg_sum_raw: float
    avg_sum_z: float
    avg_event_count: float
    avg_chan_count: float
    avg_region_count: float
    most_dominant_region: str

    power_law_fits: Dict[str, PowerLawFitResult]
    
    size_duration_beta_coeff: Dict[str, Dict]
    theoretical_size_duration_beta_coeff: Dict[str, Dict]
    deviation_from_criticality: Dict[str, Dict]

    def get(self, attribute: str):
        if hasattr(self, attribute):
            return getattr(self, attribute)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attribute}'")


class NeuronalAvalanche:
    def __init__(self, eeg_file):
        self.participant_id = eeg_file.participant_id
        self.eeg_data = eeg_file.eeg_data
        self.sampling_frequency = eeg_file.sampling_frequency
        self.channel_names = eeg_file.channel_names
        self.channel_groups = eeg_file.channel_groups
        self.segment_duration = None
        self.segmented_eeg_flag = None
        self._segmented_eeg = None
        self.region_wise = None
        self.detection_params = None
        
    def calculate_segmented_eeg(self, segment_duration: Optional[int] = None):
        self.segmented_eeg_flag = False
        if segment_duration is not None:
            self.segmented_eeg_flag = True
            self.segment_duration = segment_duration 
            # Divide the EEG signal into bins of fixed duration
            self._segmented_eeg, self.win_centers = EEGPreprocessor.get_segments(
                data=self.eeg_data, 
                sampling_frequency=self.sampling_frequency,
                window_size=self.segment_duration
            )   
    
    def detect_avalanches(self, 
                        threshold: float = 2, 
                        epoch_names: Optional[List[str]] = None, 
                        method: str = 'beggs', 
                        inter_event_interval: int = 8, 
                        bin_size: int = 2,
                        use_pearson_correlation: bool = True, 
                        correlation_threshold: float = 0.75, 
                        event_boundary: str = 'zero_crossing'):       
        """
        Detect avalanches in EEG data using the specified method.

        Parameters
        ----------
        General Parameters:
        - threshold : float, optional
            The threshold value for detecting avalanches. Default is 2.
        - method : str, optional
            The method to use for avalanche detection: 'beggs' or 'scarpetta'. Default is 'beggs'.
            
            Both methods first:
            1. Z-scores the signal amplitude on each EEG channel. 
            
            Beggs method detects over-threshold events on each EEG channel separately and:
            2. As an optional step (as per ref.1) for each event, add an event corresponding to signals on all other channels that showed correlated activity with the channel during the initially detected event.
                2-a. Set event boundaries (threshold crossing, before/after the peak, or zero-crossing before/after the peak).
                2-b. Use Pearson correlation for identifying correlated events across channels.
            3. Create a digitized signal marking the peak of each detected event.
            4. Detect avalanches as sequences of events with a maximum inter-event interval.
                
            Scarpetta method performs the following steps:
            2. Detect avalanches as periods during which at least one channel shows an absolute Z-scored amplitude above the specified threshold.

            Lasly, for each detected avalanche, store the EEG (and in case of Beggs method, also digitized EEG data) during the avalanche and flanking silence periods. The silence periods are periods with no excursions beyond the threshold on any channel.
                - This is done by storing the periods from the end of the previous avalanche to the start of the current avalanche and the end of current avalanche to the start of the next avalanche. 

        Data Parameters:
        - epoch_names : list of str, optional
            A list of epoch names to process. If None, all epochs will be processed. Default is None.

        Beggs-Specific Parameters:
        - inter_event_interval : int, optional
            The maximum distance (in ms) between events for them to be considered part of the same avalanche. Default is 8.
        - use_pearson_correlation : bool, optional
            Whether to use Pearson correlation to identify correlated events across channels. Default is True. The following parameters come into play if this is True.
        - correlation_threshold : float, optional
            The correlation threshold for identifying correlated events. Default is 0.75.
        - event_boundary : str, optional
            Method to determine the start and end of events: 'threshold_crossing' or 'zero_crossing'. Default is 'zero_crossing'.

        Notes
        -------
        'self.avalanches" is a dictionary containing:
            - 'avalanches' : A dictionary containing the raw and z-scored EEG windows of detected avalanches and their associated silence periods.
                    - 'raw': List of raw avalanche windows.
                    - 'z_scored': List of z-scored avalanche windows.
                    - 'digitized' (only for 'beggs' method): List of digitized signals during the avalanches.
            - 'avalanche_times' : A list of tuples where each tuple contains the start and end times (with the time offset) of each detected avalanche.  
            - 'silence_before' : A dictionary containing the raw and z-scored EEG windows of silence periods before each detected avalanche.
                    - 'raw': List of raw silence periods before avalanches.
                    - 'z_scored': List of z-scored silence periods before avalanches.
            - 'silence_after' : A dictionary containing the raw and z-scored EEG windows of silence periods after each detected avalanche.  
                    - 'raw': List of raw silence periods after avalanches.
                    - 'z_scored': List of z-scored silence periods after avalanches.
            - 'event_counts' (only for 'beggs' method): List of the number of events within each detected avalanche.

        References
        ----------
        Beggs method:
        1. Varley, T. et al., Beggs. J., 2020. Differential effects of propofol and ketamine on critical brain dynamics.
        2. Beggs, J., Plenz, D., 2003. Neuronal avalanches in neocortical circuits
        3. Maschke, C. et al., Blain-Moraes, S., 2023. Criticality of resting-state EEG predicts perturbational complexity and level of consciousness during anesthesia.

        Scarpetta method:
        4. Scarpetta, S. et al., 2023. Criticality of neuronal avalanches in human sleep and their relationship with sleep macro- and micro- architecture.
        """
        self.detection_params = AvalancheDetectionParams(
            threshold=threshold,
            method=method, 
            epoch_names=epoch_names, 
            inter_event_interval=inter_event_interval, 
            bin_size=bin_size,
            use_pearson_correlation=use_pearson_correlation, 
            correlation_threshold=correlation_threshold, 
            event_boundary=event_boundary
        )
        
        eeg_data = self._segmented_eeg if self.segmented_eeg_flag else self.eeg_data

        if isinstance(eeg_data, dict):
            if epoch_names is None:
                # Process all epochs if epoch_names is None
                self.avalanches = {
                    epoch_key: self._process_eeg_data(epoch_data)
                    for epoch_key, epoch_data in eeg_data.items()
                }
            else:
                # Only process specified epochs
                self.avalanches = {
                    epoch_key: self._process_eeg_data(eeg_data[epoch_key])
                    for epoch_key in self.detection_params.epoch_names if epoch_key in eeg_data
                }
        elif isinstance(eeg_data, np.ndarray):
            self.avalanches = self._process_eeg_data(eeg_data)
        else:
            raise TypeError("Unsupported data type for avalanche detection")
        
    def _process_eeg_data(self, eeg_data):
        # This function will handle both segmented and non-segmented cases
        if eeg_data.ndim == 3:  # Segmented data
            results = []
            segment_start_times = [idx * self.segment_duration * self.sampling_frequency for idx in range(eeg_data.shape[2])]
            for segment_idx, segment_start_time in enumerate(segment_start_times):
                segment_results = self._process_avalanche_data(eeg_data[:, :, segment_idx], segment_start_time)                                                      
                results.append(self._package_avalanche_data(*segment_results))
            return results
        else:  # Non-segmented data
            time_offset=0
            return self._package_avalanche_data(*self._process_avalanche_data(eeg_data, time_offset))

    def _package_avalanche_data(self, 
                                avalanches, 
                                times, 
                                silence_before, 
                                silence_after,
                                z_scored_eeg, 
                                detected_events=None,
                                event_counts=None,
                                event_details=None,
                                digitized_signal=None):
        # Helper to package the detection results into a dictionary

        results = {
            'avalanches': {
                'raw': avalanches['raw'],
                'z_scored': avalanches['z_scored']
            },
            'avalanche_times': times,
            'silence_before': {
                'raw': silence_before['raw'],
                'z_scored': silence_before['z_scored']
            },
            'silence_after': {
                'raw': silence_after['raw'],
                'z_scored': silence_after['z_scored']
            },
            'z_scored_eeg': z_scored_eeg
        }

        if self.detection_params.method == 'beggs':
            results['avalanches']['digitized'] = avalanches['digitized']
            results['detected_events'] = detected_events
            results['event_counts'] = event_counts
            results['event_details'] = event_details
            results['digitized_signal'] = digitized_signal

        return results
       
    def _process_avalanche_data(self, eeg, time_offset: int = 0):
        """
        Process EEG data to detect avalanches using the specified method.

        Parameters
        ----------
        eeg : np.ndarray
            The EEG data, which can be 2D (samples x channels) or 3D (samples x channels x trials).
        time_offset : int, optional
            An optional time offset to be added to the detected avalanche times. Default is 0.

        Returns
        -------
        tuple
            Depending on the method, returns different sets of results:
            - For 'scarpetta': (avalanches, avalanche_times, silence_before, silence_after)
            - For 'beggs': (avalanches, avalanche_times, silence_before, silence_after, event_counts, digitized_signal)
        """

        detection_params = self.detection_params
        threshold = detection_params.threshold
        method = detection_params.method 
        inter_event_interval = detection_params.inter_event_interval
        use_pearson_correlation = detection_params.use_pearson_correlation
        correlation_threshold = detection_params.correlation_threshold
        event_boundary = detection_params.event_boundary

        # Select EEG channels
        eeg_channel_indices, _ = get_eeg_channel_indices(self.channel_names, self.channel_groups)
        if eeg.ndim == 2:
            eeg = eeg[:, eeg_channel_indices]
        elif eeg.ndim == 3:
            eeg = eeg[:, eeg_channel_indices, :]

        # Calculate the z-scored EEG data
        # z_scored_eeg = EEGPreprocessor.calculate_z_score(eeg)
        z_scored_eeg = calculate_z_score_eeg(eeg, duration=60, sampling_rate=self.sampling_frequency, peak_threshold=8)

        if method == 'scarpetta':
            # Detect segments where the signal exceeds the threshold (absolute z-scored amplitude)
            over_threshold = np.any(np.abs(z_scored_eeg) > threshold, axis=1).astype(int)
            changes = np.diff(over_threshold, prepend=np.nan)

            # Determine start and end points of avalanches
            start_points = np.where(changes == 1)[0]
            end_points = np.where(changes == -1)[0]

            # Ensure there's a matching end for each start and vice versa
            if end_points.size > 0 and (start_points.size == 0 or end_points[0] < start_points[0]):
                start_points = np.insert(start_points, 0, 0)
            if start_points.size > 0 and (end_points.size == 0 or end_points[-1] < start_points[-1]):
                end_points = np.append(end_points, len(over_threshold) - 1)

        elif method == 'beggs':
            
            def enforce_types(event):
                """
                Ensure the first four items in the tuple are int, and the fifth item is float.
                """
                start, peak_index, end, chan_idx, peak_value, overthresh_dur = event
                return int(start), int(peak_index), int(end), int(chan_idx), float(peak_value), float(overthresh_dur)

            # Initialize the digitized signal and detected events set
            detected_events = []
            seen_events = set() # set to keep track of 

            for chan_idx, signal in enumerate(z_scored_eeg.T):
                abs_signal = np.abs(signal)
                peak_indices, peak_values = self._find_highest_peaks(abs_signal, threshold)
                events, overthreshold_durations = self._find_event_edges(signal, peak_indices, threshold, event_boundary)

                for (start, end), peak_index, peak_value, overthresh_dur in zip(events, peak_indices, peak_values, overthreshold_durations):
                    event = (start, peak_index, end, chan_idx, peak_value, overthresh_dur)
                    detected_events.append(enforce_types(event))
                    seen_events.add((int(peak_index), int(chan_idx)))


            # Check for correlated events across other channels
            if use_pearson_correlation:
                for event in list(detected_events):
                    start, peak_index, end, chan_idx, peak_value, _ = event
                    period = z_scored_eeg[start:end+1, chan_idx]
                    if end >= start+2: # minumum length to calculate correlation, esp. if event boundaries are set by threshold crossings 
                        period_slice = z_scored_eeg[start:end+1, :]
                        
                        correlations_matrix = np.corrcoef(period, period_slice, rowvar=False)
                        correlations = correlations_matrix[0, 1:]

                        for other_chan_idx, other_corr in enumerate(correlations):
                            if other_chan_idx != chan_idx and other_corr >= correlation_threshold:

                                other_chan_signal = z_scored_eeg[start:end+1, other_chan_idx]
                                peak_rel_index, _ = self._find_highest_peaks(np.abs(other_chan_signal), 0)
                                if len(peak_rel_index) > 0:
                                    peak_rel_index = peak_rel_index[0]
                                    other_peak_value = other_chan_signal[peak_rel_index]
                                    other_peak_idx = start + peak_rel_index

                                    if (peak_value * other_peak_value) > 0: # if the peak in the main detected event is positive include peaks in correlated channels that are also positive and vice versa
                                        new_event = (start, other_peak_idx, end, other_chan_idx, other_peak_value, -1)
                                        if (other_peak_idx, other_chan_idx) not in seen_events:
                                            detected_events.append(enforce_types(new_event))
                                            seen_events.add((int(other_peak_idx), int(other_chan_idx)))
     
            # Define a structured dtype for the array
            dtype = np.dtype([
                ('start', 'i4'), 
                ('peak_index', 'i4'), 
                ('end', 'i4'), 
                ('chan_idx', 'i4'), 
                ('peak_value', 'f4'),
                ('overthresh_dur', 'i4')
            ])
                 
            # Convert list of tuples to a structured numpy array        
            detected_events = np.array(detected_events, dtype=dtype)
            detected_events = np.sort(detected_events, order='peak_index')
            
            # Determine avalanches based on inter-event interval
            if detected_events.size > 0:
                peak_times = detected_events['peak_index']
                avalanches = np.split(detected_events, np.where(np.diff(peak_times) > (inter_event_interval / 1000.0) * self.sampling_frequency)[0] + 1)
            else:
                avalanches = []

            start_points = []
            end_points = []
            for avalanche in avalanches:
                peak_times = avalanche['peak_index']
                start_points.append(min(peak_times))
                end_points.append(max(peak_times))
                # start_points.append(min(avalanche['start']))
                # end_points.append(max(avalanche['end']))

            # Calculating the digitized signal
            digitized_signal = np.zeros_like(z_scored_eeg)
            for event in detected_events:
                peak = event['peak_index']
                chan_idx = event['chan_idx']
                digitized_signal[peak, chan_idx] = 1

        # Initialize output dictionaries
        avalanches_dict = {'raw': [], 'z_scored': [], 'digitized': [] if method == 'beggs' else None}
        avalanche_times = []
        silence_before = {'raw': [], 'z_scored': []}
        silence_after = {'raw': [], 'z_scored': []}
        max_silence_samples = int(0.5 * self.sampling_frequency)
        event_counts = [] if method == 'beggs' else None
        event_details = [] if method == 'beggs' else None # Store (time, amplitude) tuples

        # Store EEG data realted to each detected avalanche
        for i, (start, end) in enumerate(zip(start_points, end_points)):            
            
            avalanche_times.append((start + time_offset, end + time_offset))

            avalanches_dict['raw'].append(eeg[start:end + 1])
            avalanches_dict['z_scored'].append(z_scored_eeg[start:end + 1])

            if method == 'beggs':
                avalanches_dict['digitized'].append(digitized_signal[start:end + 1])

                event_count = avalanches[i].shape[0]
                event_counts.append(event_count)

                # Calculate event times and amplitudes relative to the avalanche start 
                avalanche_event_details = defaultdict(list)
                for event in avalanches[i]:
                    relative_time = event['peak_index'] - start
                    avalanche_event_details[event['chan_idx']].append((relative_time, event['peak_value']))
                event_details.append(dict(avalanche_event_details))

            raw_pre = eeg[max(end_points[i-1]+1, start-max_silence_samples):start] if i > 0 else eeg[max(0, start-max_silence_samples):start]
            z_pre = z_scored_eeg[max(end_points[i-1]+1, start-max_silence_samples):start] if i > 0 else z_scored_eeg[max(0, start-max_silence_samples):start]
            silence_before['raw'].append(raw_pre)
            silence_before['z_scored'].append(z_pre)

            next_start = start_points[i + 1] if i < len(start_points) - 1 else len(eeg)
            raw_post = eeg[end + 1:min(end + 1 + max_silence_samples, next_start)]
            z_post = z_scored_eeg[end + 1:min(end + 1 + max_silence_samples, next_start)]
            silence_after['raw'].append(raw_post)
            silence_after['z_scored'].append(z_post)

        if method == 'beggs':
            return avalanches_dict, avalanche_times, silence_before, silence_after, z_scored_eeg, detected_events, event_counts, event_details, digitized_signal
        else:
            return avalanches_dict, avalanche_times, silence_before, silence_after, z_scored_eeg

    def _find_highest_peaks(self, signal, threshold):
        """
        Find the highest peak during each period within which the signal passes the threshold.
        """
        abs_signal = np.abs(signal)
        above_threshold = (abs_signal > threshold).astype(int)

        if not np.any(above_threshold):
            return np.array([])

        changes = np.diff(above_threshold, prepend=0, append=0)
        start_indices = np.where(changes == 1)[0]
        end_indices = np.where(changes == -1)[0] - 1

        if start_indices.size > 0 and (start_indices.size == 0 or end_indices[0] < start_indices[0]):
            start_indices = np.insert(start_indices, 0, 0)
        if start_indices.size > 0 and (end_indices.size == 0 or end_indices[-1] < start_indices[-1]):
            end_indices = np.append(end_indices, len(above_threshold) - 1)

        highest_peak_indices = []
        highest_peak_values = []
        for start, end in zip(start_indices, end_indices):
            if start <= end:
                peaks, _ = find_peaks(abs_signal[start:end+1])
                if peaks.size > 0:
                    local_peaks = peaks + start
                    highest_peak = local_peaks[np.argmax(abs_signal[local_peaks])]
                    highest_peak_indices.append(highest_peak)
                    highest_peak_values.append(signal[highest_peak])

        return highest_peak_indices, highest_peak_values

    def _find_event_edges(self, signal, peaks, threshold, event_boundary):
        """Helper function to find the start and end points of events around peaks based on the chosen method."""
        events = []
        overthreshold_durations = []
        for peak_index in peaks:
            abs_signal = np.abs(signal)
            start = peak_index - np.argmax(abs_signal[:peak_index][::-1] <= threshold)
            end = peak_index + np.argmax(abs_signal[peak_index:] <= threshold)
            
            overthreshold_durations.append((end - start)/self.sampling_frequency)
            
            if event_boundary == 'zero_crossing':
                if signal[peak_index] > 0:
                    start = peak_index - np.argmax(signal[:peak_index][::-1] <= 0)
                    end = peak_index + np.argmax(signal[peak_index:] <= 0)
                else: 
                    start = peak_index - np.argmax(signal[:peak_index][::-1] >= 0)
                    end = peak_index + np.argmax(signal[peak_index:] >= 0)
            events.append((start, end))
        return events, overthreshold_durations
    
    def calculate_event_correlograms(self, max_lag: float=2.0, epoch_names=None):
        """
        Calculate the correlogram for the events detected on each channel with the events detected on other channels pooled together across all segments for each epoch.

        Parameters
        ----------
        max_lag : float, optional
            The maximum lag in seconds for which the correlogram should be computed. Default is 2 seconds.

        epoch_names : list, optional
            List of epoch names to calculate correlograms for. Default is None, which means all epochs are included.
        """
        eeg_channel_indices, _ = get_eeg_channel_indices(self.channel_names, self.channel_groups)

        correlograms_per_epoch = {}
        mean_interevent_intervals_per_epoch = {}
        max_lag_samples = int(max_lag * self.sampling_frequency)

        if epoch_names==None:
            epoch_names = list(self.avalanches.keys())
        
        for epoch_name in epoch_names:
            epoch_data = self.avalanches[epoch_name]
            all_events = defaultdict(list)
            one_channel_events = []

            # Pool all avalanches for each channel across segments
            for segment_data in epoch_data:
                detected_events = segment_data['detected_events']
                for event in detected_events:
                    all_events[event['chan_idx']].append(event['peak_index']) 
                    one_channel_events.append(event['peak_index'])

            one_channel_events = np.sort(one_channel_events)
            if len(one_channel_events) > 1:
                interevent_intervals = np.diff(one_channel_events) / self.sampling_frequency
                mean_interevent_interval = np.mean(interevent_intervals)
            else:
                mean_interevent_interval = np.nan  # Not enough events to calculate intervals

            n_channels = len(eeg_channel_indices)
            correlogram = np.zeros((n_channels, 2 * max_lag_samples + 1))

            # Determine the appropriate size for event series arrays
            max_event_index = max(max(events) for events in all_events.values()) + 1

            for chan1, events1 in all_events.items():
                # Generate binary event series
                event_series1 = np.zeros(max_event_index, dtype=int)
                event_series1[events1] = 1

                pooled_event_series = np.zeros(max_event_index, dtype=int)
                for chan2, events2 in all_events.items():
                    if chan1 != chan2:
                        pooled_event_series[events2] = 1

                # Compute the cross-correlation using scipy
                cross_correlation = correlate(event_series1, pooled_event_series, mode='full', method='auto')

                # Calculate the lags
                lags = correlation_lags(len(event_series1), len(pooled_event_series), mode="full")

                # Select relevant lags within the max_lag_samples range
                relevant_lags = np.where((lags >= -max_lag_samples) & (lags <= max_lag_samples))[0]
                correlogram[chan1, :] = cross_correlation[relevant_lags]

            correlograms_per_epoch[epoch_name] = {'correlogram': correlogram, 'lags': lags[relevant_lags]}
            mean_interevent_intervals_per_epoch[epoch_name] = mean_interevent_interval

        self.event_correlograms = correlograms_per_epoch
        self.mean_interevent_intervals = mean_interevent_intervals_per_epoch

    def calculate_avalanche_params(self, region_wise: bool = False):
        
        self.region_wise = region_wise

        if isinstance(self.avalanches, dict): # Epoched EEG   
            next_key = next(iter(self.avalanches))
            avalanche_params = {}
            if isinstance(self.avalanches[next_key], list): # Segemented Epochs
                for epoch_name, avalanche_data in self.avalanches.items():
                    print(epoch_name)
                    avalanche_params[epoch_name] = []
                    for seg_idx in range(len(avalanche_data)):
                        segment_results = self._calculate_avalanche_params_segment(avalanche_data[seg_idx])
                        avalanche_params[epoch_name].append(segment_results)
            else:# non-segmented 
                for epoch_name, avalanche_data in self.avalanches.items():
                    avalanche_params[epoch_name] = self._calculate_avalanche_params_segment(avalanche_data)           
        else: # continuous EEG (not epoched)
            avalanche_params = self._calculate_avalanche_params_segment(self.avalanches)
        self.avalanche_params = avalanche_params
            
    def _calculate_avalanche_params_segment(self, avalanche_data: Dict[str, Any]) -> AvalancheParams:

        _, eeg_channel_names = get_eeg_channel_indices(self.channel_names, self.channel_groups)
        method = self.detection_params.method

        av_times_all = avalanche_data['avalanche_times']

        raw_av_all = avalanche_data['avalanches']['raw']
        z_av_all = avalanche_data['avalanches']['z_scored']

        silence_before = avalanche_data['silence_before']['raw']
        silence_after = avalanche_data['silence_after']['raw']

        num_av = len(raw_av_all)

        duration = np.full((num_av,), np.nan)
        sum_raw = np.full((num_av,), np.nan)
        sum_z = np.full((num_av,), np.nan)
        min_silence = np.full((num_av,), np.nan)
        chan_count = np.full((num_av,), np.nan)
        region_count = np.full((num_av,), np.nan)
        event_count = np.full((num_av,), np.nan)
        dominant_region = []

        if method == 'scarpetta':
            for av_idx, (av_times, raw_av, z_av, pre, post) in enumerate(zip(av_times_all, raw_av_all, z_av_all, silence_before, silence_after)):
                
                # Avalanche duration
                if av_times[1]==av_times[0]:
                    duration_samples = 1
                else: 
                    duration_samples = (av_times[1] - av_times[0])

                duration_ms = (duration_samples/self.sampling_frequency) * 1000
                duration[av_idx] = int(np.ceil(duration_ms / self.detection_params.bin_size))

                min_silence[av_idx] = min(len(pre), len(post))  

                # Avalanche size as accumulated sum of absolute amplitudes over the threshold over all channels
                above_threshold = np.abs(z_av) > self.detection_params.threshold
                sum_raw[av_idx] = np.sum(np.abs(raw_av)[above_threshold])
                sum_z[av_idx] = np.sum(np.abs(z_av)[above_threshold])

                # Number of active channels during each avalanche 
                chan_count[av_idx] = np.sum(np.any(above_threshold, axis=0))
                
                # Calculate region-wise sum
                region_sum = {}
                for region_name, region_channels in self.channel_groups.items():
                    region_sum[region_name] = 0

                for i, channel in enumerate(eeg_channel_names):
                    if np.any(above_threshold[:, i]):
                        for region_name, region_channels in self.channel_groups.items():
                            if channel in region_channels:
                                region_sum[region_name] += np.sum(np.abs(z_av[:, i])) 

                # Calcualte region count
                region_count[av_idx] = sum(1 for sum_val in region_sum.values() if sum_val > 0)

                # Determine dominant region 
                if region_sum:
                    dominant_region.append(max(region_sum, key=region_sum.get))
                else:
                    dominant_region.append(None)
        elif method == 'beggs':
            event_details = avalanche_data['event_details']
            for av_idx, (av_times, av_event_detail, pre, post) in enumerate(zip(av_times_all, event_details, silence_before, silence_after)):
                
                # Avalanche duration
                if av_times[1]==av_times[0]:
                    duration_samples = 1
                else: 
                    duration_samples = (av_times[1] - av_times[0])

                duration_ms = (duration_samples/self.sampling_frequency) * 1000
                duration[av_idx] = int(np.ceil(duration_ms / self.detection_params.bin_size))
                
                min_silence[av_idx] = min(len(pre), len(post))

                # Number of events during each avalanche
                event_count[av_idx] = len(av_event_detail)
                
                # Summation of the peak amplitude of the detected events pertaining to each avalanche
                peak_values = []
                for event_list in av_event_detail.values():
                    for _, peak_value in event_list:  # Unpack each tuple within the list
                        peak_values.append(peak_value)
                sum_z[av_idx] = np.sum(peak_values)
                
                # Number of active channels during each avalanche
                chan_indices = set(av_event_detail.keys())
                chan_count[av_idx] = len(chan_indices)
                
                # Regions active during each avalanche
                region_sum = {}
                for region_name, region_channels in self.channel_groups.items():
                    region_sum[region_name] = 0

                active_channels = [eeg_channel_names[i] for i in chan_indices]
                for i, channel in enumerate(active_channels):
                    for region_name, region_channels in self.channel_groups.items():
                        if channel in region_channels:
                            region_sum[region_name] += 1
                
                # Calculate region count
                region_count[av_idx] = sum(1 for sum_val in region_sum.values() if sum_val > 0)

                # Determine dominant region
                if region_sum:
                    dominant_region.append(max(region_sum, key=region_sum.get))
                else:
                    dominant_region.append(None)

        # Calculate averages of the avalanche parameters across avalanches within an EEG segment
        occurrence_frequency = num_av/self.segment_duration

        # Filtering the avalanches with outlier size and minimum number of events
        sum_z = np.array(sum_z)
        _, accepted_sum_z = remove_outliers(sum_z)

        event_count = np.array(event_count)
        have_min_event_counts = event_count >= 2

        accepted_avalanches = np.where(np.logical_and(accepted_sum_z, have_min_event_counts))[0]

        sum_z = sum_z[accepted_avalanches]
        avg_sum_z = np.mean(sum_z)

        sum_raw = np.array(sum_raw)
        sum_raw = sum_raw[accepted_avalanches]
        avg_sum_raw = np.mean(sum_raw) if method == 'scarpetta' else None
        
        duration = np.array(duration)
        duration = duration[accepted_avalanches]
        avg_duration = np.mean(duration)

        event_count = event_count[accepted_avalanches]
        avg_event_count = np.mean(event_count)

        chan_count = np.array(chan_count)
        chan_count = chan_count[accepted_avalanches]
        avg_chan_count = np.mean(chan_count) 
        
        region_count = np.array(region_count)
        region_count = region_count[accepted_avalanches]
        avg_region_count = np.mean(region_count) 

        region_counter = Counter(dominant_region)
        most_dominant_region = region_counter.most_common(1)[0][0] if region_counter else None

        avalanche_params = AvalancheParams(
            duration=duration.tolist(),
            sum_raw=sum_raw.tolist(), 
            sum_z=sum_z.tolist(), 
            event_count=event_count.tolist(),
            min_silence=min_silence.tolist(), 
            chan_count=chan_count.tolist(), 
            region_count=region_count.tolist(), 
            dominant_region=dominant_region,
            occurrence_frequency=occurrence_frequency, 
            avg_duration=avg_duration, 
            avg_sum_raw=avg_sum_raw, 
            avg_sum_z=avg_sum_z, 
            avg_event_count = avg_event_count,
            avg_chan_count=avg_chan_count, 
            avg_region_count=avg_region_count, 
            most_dominant_region = most_dominant_region,
            power_law_fits={},
            size_duration_beta_coeff={},
            theoretical_size_duration_beta_coeff={},
            deviation_from_criticality={}
        )

        avalanche_params.power_law_fits = self.power_law_analysis(avalanche_params)
        avalanche_params.size_duration_beta_coeff = self.size_duration_relationship(avalanche_params)
        avalanche_params.theoretical_size_duration_beta_coeff = self.calculate_theoretical_beta(avalanche_params)
        avalanche_params.deviation_from_criticality = self.calculate_deviation_from_criticality(avalanche_params)

        return avalanche_params
    
    def power_law_analysis(self, avalanche_params: AvalancheParams) -> Dict[str, PowerLawFitResult]:
        """
        Analyze power-law distribution of various avalanche parameters for each region and overall.

        Returns:
        results (dict): A nested dictionary where keys are region names or "overall",
                        and values are dictionaries with parameter names and PowerLawFitResult objects.
        """
        results = {}

        params_to_analyze = {
            'duration': avalanche_params.duration,
            'sum_z': avalanche_params.sum_z,
            'chan_count': avalanche_params.chan_count,
            'event_count': avalanche_params.event_count
        }
        if self.detection_params.method == 'scarpetta':
            params_to_analyze['sum_raw'] = avalanche_params.sum_raw

        # Configuration for generating xmins for each parameter
        # fit_range_config = {
        #     'duration': (1, 20),
        #     'sum_raw': (5e-5, 1e-2),
        #     'sum_z': (4, 30),
        #     'chan_count': (2, 16), 
        #     'event_count': (2, 16) 
        # }

        # parameter_range= {"alpha": [1., 4.]}

        num_values = 20  # number of samples to use for calculation of (xmin, xmax) pairs

        regions = set(avalanche_params.dominant_region)
        regions.discard(None)
        analysis_regions = regions.union({'overall'}) if self.region_wise else {'overall'}
        
        for region in analysis_regions:
            region_results = {}

            for param, data in params_to_analyze.items():

                if len(data) == 0 or np.all(np.isnan(data)):
                    continue

                # Filter data based on region
                if region != 'overall':
                    region_data = np.array([d for d, r in zip(data, avalanche_params.dominant_region) if r == region])
                else:
                    region_data = np.array(data)

                region_data = region_data[~np.isnan(region_data) & ~np.isinf(region_data)]
                if len(region_data) == 0:
                    continue

                # start_value, stop_value = fit_range_config[param]
                pla = PowerLawAnalyzer(region_data)
                region_results[param] = pla.compute_power_law_fits(num_values=num_values)
                # region_results[param] = pla.compute_power_law_fits(start_value, stop_value, num_values)

            results[region] = region_results

        return results if self.region_wise else results['overall']

    def size_duration_relationship(self, avalanche_params: AvalancheParams):
        """
        Analyze the size-duration relationship of avalanches, including region-specific analysis.

        Returns:
            results (dict): A nested dictionary where keys are region names or "overall",
                            and values are dictionaries with size parameter names, 
                            and nested dictionaries containing beta coefficient results.
        """

        # def _calculate_beta_coefficients(durations, sizes):
        #     """Helper function to calculate regression coefficients and p-values"""
        #     log_durations = np.log10(durations)
        #     log_sizes = np.log10(sizes)
        #     log_durations_with_const = sm.add_constant(log_durations)
        #     model = sm.OLS(log_sizes, log_durations_with_const)
        #     results = model.fit()
        #     try:
        #         return results.params[1], results.pvalues[1]
        #     except:
        #         return np.nan, np.nan

        # def _calculate_beta_coefficients(durations, sizes):
        #     """Helper function to calculate regression coefficients and p-values"""
        #     # Convert data to pandas DataFrame
        #     data = pd.DataFrame({'duration': durations, 'size': sizes})

        #     # Group by 'duration' and calculate the mean 'size' for each group
        #     grouped_data = data.groupby('duration').mean().reset_index()

        #     # Extract the grouped 'size' and averaged 'duration'
        #     grouped_sizes = grouped_data['size']
        #     grouped_durations = grouped_data['duration']

        #     # Perform log transformation
        #     log_durations = np.log10(grouped_durations)
        #     log_sizes = np.log10(grouped_sizes)

        #     # Perform linear regression
        #     log_durations_with_const = sm.add_constant(log_durations)
        #     model = sm.OLS(log_sizes, log_durations_with_const)
        #     results = model.fit()
        #     try:
        #         beta = results.params['duration']
        #         p_value = results.pvalues['duration']
        #         intercept = results.params['const']
        #         return beta, p_value, intercept
        #     except:
        #         return np.nan, np.nan, np.nan
        def _calculate_beta_coefficients(durations, sizes):
            """Helper function to calculate regression coefficients and p-values with weighted least squares"""
            # Convert data to pandas DataFrame
            data = pd.DataFrame({'duration': durations, 'size': sizes})

            # Group by 'duration' and calculate the mean 'size' and count for each group
            grouped_data = data.groupby('duration').agg(size_mean=('size', 'mean'), count=('size', 'size')).reset_index()

            # Extract the grouped 'size', averaged 'duration', and counts
            grouped_sizes = grouped_data['size_mean']
            grouped_durations = grouped_data['duration']
            weights = grouped_data['count']  # Use count as the weights

            # Perform log transformation
            log_durations = np.log10(grouped_durations)
            log_sizes = np.log10(grouped_sizes)

            # Perform weighted least squares regression
            log_durations_with_const = sm.add_constant(log_durations)
            model = sm.WLS(log_sizes, log_durations_with_const, weights=weights)
            results = model.fit()
            try:
                beta = results.params['duration']
                p_value = results.pvalues['duration']
                intercept = results.params['const']
                return beta, p_value, intercept
            except:
                return np.nan, np.nan, np.nan

        def _calculate_beta_coefficients_in_specified_range(durations, sizes, xmin_duration, xmax_duration, xmin_size, xmax_size):
            """Helper function to first filter the duration and size data and then calculate the beta coefficient"""
            
            # av_subset_idx = np.where(
            #     (np.array(durations) >= xmin_duration) &
            #     (np.array(durations) <= xmax_duration) &
            #     (np.array(sizes) >= xmin_size) &
            #     (np.array(sizes) <= xmax_size)
            # )[0]
            
            # if len(av_subset_idx) > 0:
            curr_durations = np.array(durations)#[av_subset_idx]
            curr_sizes = np.array(sizes)#[av_subset_idx]
            
            return _calculate_beta_coefficients(curr_durations, curr_sizes)
            # else:
                # return np.nan, np.nan, np.nan

        analysis_results = {}

        size_params_to_analyze = {
            'sum_z': avalanche_params.sum_z,
            'chan_count': avalanche_params.chan_count,
            'event_count': avalanche_params.event_count
        }
        if self.detection_params.method == 'scarpetta':
            size_params_to_analyze['sum_raw'] = avalanche_params.sum_raw

        durations = avalanche_params.duration
        duration_fits = avalanche_params.power_law_fits['overall']['duration'] if self.region_wise else avalanche_params.power_law_fits['duration']
        num_ranges_duration = len(duration_fits.xmins)

        regions = set(avalanche_params.dominant_region)
        regions.discard(None)
        analysis_regions = regions.union({'overall'}) if self.region_wise else {'overall'}

        for region in analysis_regions:
            region_analysis_results = {}

            for size_param, sizes in size_params_to_analyze.items():
                if len(sizes) == 0 or np.all(np.isnan(sizes)):
                    continue

                if region != 'overall':
                    region_durations = [d for d, r in zip(durations, avalanche_params.dominant_region) if r == region]
                    region_sizes = [s for s, r in zip(sizes, avalanche_params.dominant_region) if r == region]
                    region_duration_fits = avalanche_params.power_law_fits[region]['duration']
                    region_size_fits = avalanche_params.power_law_fits[region][size_param]
                else:
                    region_durations = durations
                    region_sizes = sizes
                    region_duration_fits = duration_fits
                    region_size_fits = avalanche_params.power_law_fits['overall'][size_param] if self.region_wise else avalanche_params.power_law_fits[size_param]

                num_ranges_size = len(region_size_fits.xmins)            

                # initialize 2D arrays for beta_coeff and p_value
                beta_coeff = np.full((num_ranges_duration, num_ranges_size), np.nan)
                p_value = np.full((num_ranges_duration, num_ranges_size), np.nan)
                intercept = np.full((num_ranges_duration, num_ranges_size), np.nan)
                
                for i_xmin_duration, (xmin_duration, xmax_duration) in enumerate(zip(region_duration_fits.xmins, region_duration_fits.xmaxs)):
                    for i_xmin_size, (xmin_size, xmax_size) in enumerate(zip(region_size_fits.xmins, region_size_fits.xmaxs)):
                        
                        (beta_coeff[i_xmin_duration][i_xmin_size], 
                         p_value[i_xmin_duration][i_xmin_size],
                         intercept[i_xmin_duration][i_xmin_size]) = _calculate_beta_coefficients_in_specified_range(
                             region_durations, 
                             region_sizes, 
                             xmin_duration, 
                             xmax_duration, 
                             xmin_size, 
                             xmax_size
                        )
                                
                # analyzing using optimum xmins
                optimum_xmin_duration = region_duration_fits.xmin
                optimum_xmax_duration = region_duration_fits.xmax
                optimum_xmin_size = region_size_fits.xmin
                optimum_xmax_size = region_size_fits.xmax

                optimum_xmin_beta_coeff, optimum_xmin_p_value, optimum_xmin_intercept = _calculate_beta_coefficients_in_specified_range(
                    region_durations, 
                    region_sizes, 
                    optimum_xmin_duration, 
                    optimum_xmax_duration, 
                    optimum_xmin_size, 
                    optimum_xmax_size
                )

                region_analysis_results[size_param] = {
                    "beta": {
                        'coeff': beta_coeff,
                        'p_value': p_value,
                        'intercept': intercept
                    },
                    "optimum_beta": {
                        'coeff': optimum_xmin_beta_coeff,
                        'p_value': optimum_xmin_p_value,
                        'intercept': optimum_xmin_intercept
                    }
                }
            analysis_results[region] = region_analysis_results
            
        return analysis_results if self.region_wise else analysis_results['overall']
    
    def calculate_theoretical_beta(self, avalanche_params: AvalancheParams):
        """
        Calculate the theoretical beta coefficient (or beta prime) as 
        (alpha_duration - 1)/(alpha_size - 1).

        A distribution of beta prime can be calculated by taking any two alphas from 
        the alternative lists of values corresponding to duration and size, resulting from different xmins

        OR

        a single beta prime can be calculated by considering the optimum alphas from the two lists.

        Parameters:
        - avalanche_params (AvalancheParams): An object containing the alpha values for duration and size, as well as their optimum values.

        Returns:
        - A nested dictionary where keys are region names or "overall",
            and values are dictionaries with size parameter names,
            and nested dictionaries containing beta prime results.
        """
        analysis_results = {}

        regions = set(avalanche_params.dominant_region)
        regions.discard(None)

        analysis_regions = regions.union({'overall'}) if self.region_wise else {'overall'}

        for region in analysis_regions:
            region_analysis_results = {}

            if region == 'overall':
                region_duration_data = avalanche_params.power_law_fits['overall']['duration'] if self.region_wise else avalanche_params.power_law_fits['duration']
                region_size_data = {
                    'sum_z': avalanche_params.power_law_fits['overall']['sum_z'] if self.region_wise else avalanche_params.power_law_fits['sum_z'],
                    'chan_count': avalanche_params.power_law_fits['overall']['chan_count'] if self.region_wise else avalanche_params.power_law_fits['chan_count'],
                    'event_count': avalanche_params.power_law_fits['overall']['event_count'] if self.region_wise else avalanche_params.power_law_fits['event_count']
                }
                if self.detection_params.method == 'scarpetta':
                    region_size_data['sum_raw']= avalanche_params.power_law_fits['overall']['sum_raw'] if self.region_wise else avalanche_params.power_law_fits['sum_raw']
            else:
                region_duration_data = avalanche_params.power_law_fits.get(region, {}).get('duration', None)
                if region_duration_data is None:
                    continue
                region_size_data = {
                    'sum_z': avalanche_params.power_law_fits.get(region, {}).get('sum_z', None),
                    'chan_count': avalanche_params.power_law_fits.get(region, {}).get('chan_count', None),
                    'event_count': avalanche_params.power_law_fits.get(region, {}).get('event_count', None)
                }
                if self.detection_params.method == 'scarpetta':
                    region_size_data['sum_raw']= avalanche_params.power_law_fits.get(region, {}).get('sum_raw', None)

            alphas_duration = region_duration_data.alphas
            optimum_alpha_duration = region_duration_data.alpha

            if alphas_duration and all(size_data is not None for size_data in region_size_data.values()):           
                for size_param, size_data in region_size_data.items():
                    alphas_size = size_data.alphas
                    optimum_alpha_size = size_data.alpha

                    if alphas_size:
                        # initialize a 2D array for beta primes
                        beta_primes = np.zeros((len(alphas_duration), len(alphas_size)))
                        
                        for i, alpha_duration in enumerate(alphas_duration):
                            for j, alpha_size in enumerate(alphas_size):
                                if alpha_size != 1:
                                    beta_primes[i][j] = (alpha_duration - 1) / (alpha_size - 1)
                                else:
                                    beta_primes[i][j] = float('inf')

                        # Calculate the optimum beta prime
                        if optimum_alpha_size != 1:
                            optimum_beta_prime = (optimum_alpha_duration - 1) / (optimum_alpha_size - 1)
                        else:
                            optimum_beta_prime = float('inf')

                        region_analysis_results[size_param] = {
                            'beta_prime': beta_primes,
                            'optimum_beta_prime': optimum_beta_prime
                        }
                    else:
                        warn(f'No avalanche size {size_param} data for region {region}. Returning empty beta prime array')
                        region_analysis_results[size_param] = {}

            else:
                warn(f'No avalanche duration data for region {region}. Returning empty beta prime array')

            analysis_results[region] = region_analysis_results

        return analysis_results if self.region_wise else analysis_results['overall']
    
    def calculate_deviation_from_criticality(self, avalanche_params: AvalancheParams):
        """
        Calculate the absolute difference between the beta coefficient (obtained by linear regression analysis of size versus duration)
        and theoretical beta coefficient or beta prime by taking the ratio of alpha values of the powerlaw PDFs fitted to
        distributions of size and duration. 

        Returns:
        A nested dictionary where keys are region names or "overall",
        and values are dictionaries with size parameter names,
        and nested dictionaries containing dcc results. 
        """
        analysis_results = {}
        
        beta_data = avalanche_params.size_duration_beta_coeff
        beta_prime_data = avalanche_params.theoretical_size_duration_beta_coeff

        size_params = beta_prime_data['overall'].keys() if self.region_wise else beta_prime_data.keys()

        if not beta_data or not beta_prime_data:
            warn('No beta or beta prime data in avalanche_params.')
            return analysis_results

        regions = set(avalanche_params.dominant_region)
        regions.discard(None)

        analysis_regions = regions.union({'overall'}) if self.region_wise else {'overall'}

        for region in analysis_regions:
            region_analysis_results = {}

            for size_param in size_params:
                if region == 'overall':
                    region_beta = beta_data['overall'].get(size_param, {}) if self.region_wise else beta_data.get(size_param, {})
                    region_beta_prime = beta_prime_data['overall'].get(size_param, {}) if self.region_wise else beta_prime_data.get(size_param, {})
                else:
                    region_beta = beta_data.get(region, {}).get(size_param, {})
                    region_beta_prime = beta_prime_data.get(region, {}).get(size_param, {})

                if region_beta and region_beta_prime:
                    beta = region_beta['beta']['coeff']
                    optimum_beta = region_beta['optimum_beta']['coeff']

                    beta_prime = np.array(region_beta_prime['beta_prime'])
                    optimum_beta_prime = region_beta_prime['optimum_beta_prime']

                    dcc = np.abs(beta_prime - beta)
                    dcc_optimum = np.abs(optimum_beta_prime - optimum_beta)
                    
                    region_analysis_results[size_param] = {
                        'dcc': dcc,
                        'dcc_optimum': dcc_optimum
                    }
                else:
                    warn(f"No beta or beta prime data for region '{region}' and size parameter '{size_param}'.")
                    region_analysis_results[size_param] = {}

            analysis_results[region] = region_analysis_results
    
        return analysis_results if self.region_wise else analysis_results['overall']