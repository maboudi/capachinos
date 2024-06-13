import mne
from src.eeg_analysis.utils.helpers import create_mne_raw_from_data 

class PowerSpectralAnalysis:
    def __init__(self, eeg_data):
        """
        Initialize the PowerSpectralAnalysis with preprocessed EEG data.

        Parameters:
        preprocessed_data
        """
        self.participant_id = eeg_data.participant_id
        self.preprocessed_data = eeg_data.eeg_with_rejected_noisy_segments
        self.channel_names = eeg_data.channel_names
        self.sampling_frequency = eeg_data.sampling_frequency

        if hasattr(eeg_data, 'ds_sampling_frequency'):
            self.ds_sampling_frequency = eeg_data.ds_sampling_frequency
        else:
            self.ds_sampling_frequency = None 
        
        self.ds_sampling_frequency = eeg_data.ds_sampling_frequency
        self.whitened_signal = None
        self.tfr = None
        self.spectrogram = None
        self.group_spectrogram = None
        self.emergence_trajectory = None

    def whiten_signal(self):
        """
        Apply whitening to the signal.
        """
        # Placeholder for actual signal whitening logic
        self.whitened_signal = self.preprocessed_data  # Replace with actual whitening operation

    def calculate_time_frequency_map(self, method = 'multitaper', select_channels=None):
        """
        Calculate power spectral density of the signal using multi-taper method
        """
        ch_names = self.channel_names
        sfreq = self.ds_sampling_frequency if self.ds_sampling_frequency is not None else self.sampling_frequency
        
        data_to_process = self.preprocessed_data

        def calculate_time_frequency_map_epoch(epoch, ch_names, sfreq):

            # Create an MNE Info object with the properties of your data
            raw = create_mne_raw_from_data(epoch, ch_names, sfreq)

            # frequencies = np.logspace(*np.log10([1, 45]), num=40)  # Define the range of frequencies
            frequencies = np.arange(0.5, 45.5, 0.5)
            n_cycles = frequencies / 2.  # Different number of cycle per frequency

            if method.lower() is 'multitaper':
                # Multitaper analysis
                time_bandwidth = 3 # n_tapers = int(time_bandwidth * 2 - 1)

                tfr = mne.time_frequency.tfr_multitaper(
                    raw, 
                    freqs=frequencies, 
                    n_cycles=n_cycles, 
                    time_bandwidth=time_bandwidth, 
                    return_itc=False
                )
            elif method.lower() is 'morlet':            
                # Morlet wavelets analysis
                tfr = mne.time_frequency.tfr_morlet(
                    raw, 
                    freqs=frequencies, 
                    n_cycles=n_cycles,
                    return_itc=False, 
                    decim=3, 
                    n_jobs=1
                )
            elif method.lower() is 'spectrogram':
                tfr = None
            else:
                raise ValueError("Unsupported time-frequency representation method: {}".format(method))
            return tfr

        if isinstance(data_to_process, dict):
            self.tfr = {epoch_name: calculate_time_frequency_map_epoch(epoch_data, ch_names, sfreq)
                        for epoch_name, epoch_data in data_to_process.items()}
        else:
            self.tfr = calculate_time_frequency_map_epoch(data_to_process)   

        self.tfr = None

    def calculate_group_level_spectrogram(self):
        """
        Calculate group-level spectrogram for the signal.
        """
        # Placeholder for actual group-level spectrogram calculation logic
        self.group_spectrogram = {}  # Replace with actual group-level spectrogram calculation

    def calculate_emergence_trajectory(self):
        """
        Calculate power for each frequency band.
        """
        # Placeholder for actual spectral parameter calculation logic
        self.emergence_trajectory = {}  # Replace with actual spectral parameter calculation

    def calculate_spectral_dynamics(self):
        """
        Identify states in the spectral feature space, 
        calculate a Markov transition probablity matrix, 
        and characterize the emergence period in terms of transition between distinct states 
        """
        pass

    def display_power_spectra(self):
        """
        Visualize power spectra.
        """
        # Placeholder for actual visualization logic
        pass  # Replace with actual visualization code



"""
# Example usage:
# Assuming preprocessed_data is an object with a participant_id attribute and preprocessed EEG data
preprocessed_data = type('PreprocessedData', (object,), {'participant_id': 1, 'data': 'Preprocessed EEG data'})()  # Dummy preprocessed data object
psa = PowerSpectralAnalysis(preprocessed_data)
psa.whiten_signal()
psa.calculate_power_spectral()
psa.calculate_group_level_spectrogram()
psa.display_power_spectra()
psa.calculate_spectral_parameters()

print(psa.whitened_signal)
print(psa.power_spectral_density)
print(psa.group_spectrogram)
print(psa.spectral_parameters)
"""


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


"""
# Example usage:
data_per_individual = {}  # Dictionary to store data for each individual
num_participants = 10
window_size = 20
frequency_intervals = [(0, 4), (4, 8), (8, 12), (12, 30)]

# Generate random data for each participant
for i in range(num_participants):
    data_per_individual[i+1] = np.random.rand(100, 32)  # Example data

# Create MetastableSpectralDynamicAnalysis objects for each individual
msda_per_individual = {}
for participant_id, data in data_per_individual.items():
    msda_per_individual[participant_id] = MetastableSpectralDynamicAnalysis(participant_id, data)

# Perform analysis for each individual
for msda in msda_per_individual.values():
    msda.construct_spectral_power_vectors(window_size, frequency_intervals)
    msda.remove_outlier_windows()

# Concatenate data whenever needed
concatenated_vectors = np.concatenate([msda.pca_vectors for msda in msda_per_individual.values()], axis=0)

# Apply PCA and remaining analysis steps for individual or concatenated data
# Example for individual data
individual_msda = msda_per_individual[1]  # Assuming we want to analyze individual data for participant 1
individual_msda.apply_pca(num_components=10)
individual_msda.categorize_clusters(num_clusters=5)

# Example for concatenated data
pca = PCA(n_components=10)
pca.fit(concatenated_vectors)
pca_vectors_concatenated = pca.transform(concatenated_vectors)
kmeans = KMeans(n_clusters=5)
cluster_labels_concatenated = kmeans.fit_predict(pca_vectors_concatenated)
cluster_quality_concatenated = silhouette_score(pca_vectors_concatenated, cluster_labels_concatenated)
"""