import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import zscore
import powerlaw
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


class CriticalityAnalysis:
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


class NeuronalAvalanche:
    def __init__(self, eeg_signal, bin_duration):
        self.eeg_signal = eeg_signal
        self.bin_duration = bin_duration
        self.z_signal = zscore(eeg_signal.signal)
        self.num_bins = int(np.ceil(eeg_signal.duration / bin_duration))

    def bin_signal(self):
        # Divide the EEG signal into bins of fixed duration
        binned_signal = np.array_split(self.z_signal, self.num_bins)
        # Some bins might be shorter than bin_duration. If exact-duration bins are needed, padding could be applied
        return binned_signal

    def detect_avalanches(self, threshold=2):
        # Creating bins of z-scored EEG signal
        binned_signal = self.bin_signal()
        # Finding bins that exceed the threshold
        over_threshold = [np.any(bin_i > threshold) for bin_i in binned_signal]
        
        # Tracking changes (going above or below threshold)
        changes = np.diff(over_threshold).nonzero()[0]
        
        avalanches = []
        # Ensure that we start with a positive threshold crossing
        start_idx = 0 if over_threshold[0] else changes[0] + 1

        for i in range(start_idx, len(changes), 2):
            start = changes[i]
            end = changes[i + 1] if i + 1 < len(changes) else len(over_threshold)
            
            # Calculating the avalanche window
            avalanche_window = np.concatenate(binned_signal[start:end + 1])
            avalanches.append(avalanche_window)
            
        return avalanches

    def calculate_avalanche_metrics(self, avalanches):
        duration = [len(a) for a in avalanches]
        size = [np.sum(a) for a in avalanches]
        return duration, size

    def power_law_analysis(self, durations, sizes):
        fit_results = {}
        for data, name in ((durations, 'duration'), (sizes, 'size')):
            fit = powerlaw.Fit(np.array(data) + 1, xmin=np.min(data))
            fit_results[name] = {
                'alpha': fit.power_law.alpha,
                'D': fit.power_law.D}
        return fit_results

    def size_duration_relationship(self, durations, sizes):
        log_sizes = np.log10(sizes)
        log_durations = np.log10(durations)
        
        coeffs = np.polyfit(log_durations, log_sizes, deg=1)
        beta = coeffs[0]
        return beta


# Utility functions that may be used across classes
def binarize_phase(phase):
    return 1 if phase > 0 else 0


"""
Example usage:

# Simulate EEG data using a random process (for illustrative purposes)
def simulate_eeg_data(duration_seconds, sampling_rate, num_channels):
    time = np.linspace(0, duration_seconds, int(duration_seconds * sampling_rate))
    eeg_data = np.random.randn(num_channels, len(time))
    return eeg_data, time

duration_seconds = 20  # 20 seconds of data
sampling_rate = 256  # Sampling rate of the signals
num_channels = 10  # Number of EEG channels

# Simulate EEG signals
simulated_eeg_data, time = simulate_eeg_data(duration_seconds, sampling_rate, num_channels)

# Filter and autocorrelation analysis
frequency_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    # Add more frequency bands if needed
}

eeg_signals = [EEGSignal(simulated_eeg_data[i, :], sampling_rate) for i in range(num_channels)]

# Print example of autocorrelation function calculation for the first EEG channel
criticality_analyzer = CriticalityAnalysis(eeg_signals[0], frequency_bands)
acf_analysis_results = criticality_analyzer.perform_acf_analysis()

# Example for plotting ACF(1) heatmap for the first EEG channel
criticality_analyzer.plot_acf_heatmap(acf_analysis_results)

# Avalanche detection example for first EEG channel
bin_duration = 0.01  # 10ms bins
neuronal_avalanche_processor = NeuronalAvalanche(eeg_signals[0], bin_duration)
avalanches = neuronal_avalanche_processor.detect_avalanches()

print(f"Detected avalanches: {len(avalanches)}")

"""