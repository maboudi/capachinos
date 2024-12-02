from src.eeg_analysis.utils.helpers import create_mne_raw_from_data 
import mne
import numpy as np

class ConnectivityAnalysis:
    def __init__(self, eeg_file):
        """
        Initialize the ConnectivityAnalysis with EEG data.

        Parameters:
        eeg_data: The EEG data for analysis.
        """
        self.participant_id = eeg_file.participant_id
        self.eeg_data = eeg_file.eeg_data
        self.channel_names = eeg_file.channel_names
        self.channel_groups = eeg_file.channel_groups

        self.wpli_matrix = None
        self.connectivity_over_time = None
        self.inter_regional_connectivity = None
        self.frequency_band_wpli = None
        self.connectivity_state_index_time_series = None
        self.transition_matrix_counts = None
        self.transition_matrix_probabilities = None

    def calculate_cross_spectral_density(self, method='multitaper', select_channels=None, select_epochs=None):
        """
        Calculate cross-spectral density (CSD) for the EEG data.

        This method selects the appropriate EEG data (either raw or previously 
        whitened data, if available) and calculates the CSD using the specified method.

        Parameters:
        - method (str): Specifies the method to compute the cross spectral density. Options include
                        'multitaper', 'morlet', and 'fourier'. Default is 'multitaper'.
        - select_channels (list): List of channel names or indices to include in the analysis.
                                  If None, all channels are used.
        - select_epochs (list): List of epoch names to include in the analysis. 
                                If None, all epochs are processed.
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
                self.csd[epoch_name], self.channel_names = self._calculate_epoch_csd(epoch_data, method, select_channels)
        
        # If data_source in continuous EEG data 
        else:
            self.csd, self.channel_names = self._calculate_epoch_csd(data_source, method, select_channels)

    def _calculate_epoch_csd(self, epoch_data, method, select_channels):
        
        # The following method is in power_spectral.py
        ch_names, ch_types, epoch = self._select_channels_and_adjust_data(epoch_data, select_channels)
        
        raw = create_mne_raw_from_data(
            data=epoch, 
            channel_names=ch_names, 
            sampling_frequency=self.sampling_frequency, 
            ch_types=ch_types
        )

        if method.lower()=='multitaper':
            csd = self._calculate_multitaper_csd(raw)
        elif method.lower()=='morlet':
            csd = self._calculate_morlet_csd(raw)            
        elif method.lower()=='fourier':
            csd = self._calculate_fourier_csd(raw)
        else:
            raise ValueError("Unsupported time-frequency representation method: {}".format(method))
        
        return csd, ch_names
    
    def _calculate_multitaper_csd(self, raw):
        return mne.time_frequency.csd_multitaper(
            raw, 
            fmin= np.min(self.frequencies), 
            fmax = np.max(self.frequencies),
            adaptive=True,
            verbose=False
        )
    
    def _calculate_morlet_csd(self, raw):
        return mne.time_frequency.csd_morlet(
            raw, 
            self.frequencies, 
            decim=10, 
            verbose=False
        )
    
    def _calculate_fourier_csd(self, raw):
        return mne.time_frequency.csd_fourier(
            raw, 
            fmin= np.min(self.frequencies), 
            fmax = np.max(self.frequencies),
            verbose=False
        )
    
    def ft_connectivity_wpli(inputdata, dojack=False, debias=False, isunivariate=False, cumtapcnt=None, feedback='none'):
        """
        Compute the weighted phase lag index (wpli) from a data matrix containing the cross-spectral density.
        
        Parameters:
        - inputdata: array containing cross-spectral densities organized as:
                    Repetitions x Channel x Channel (x Frequency) (x Time) -or-
                    Repetitions x Channelcombination (x Frequency) (x Time)
                    Alternatively, can contain Fourier coefficients organized as:
                    Repetitions_tapers x Channel (x Frequency) (x Time)
        - dojack: boolean, compute a variance estimate based on leave-one-out
        - debias: boolean, compute debiased wpli or not
        - isunivariate: boolean, compute CSD on-the-fly (saves memory with many trials)
        - cumtapcnt: array, cumulative taper counter defining repetitions
        - feedback: type of feedback showing progress of computation ('none', 'text', 'textbar', etc.)

        Returns:
        - wpli: Weighted Phase Lag Index
        - v: leave-one-out variance estimate (optional)
        - n: number of repetitions in the input data

        The function computes the weighted phase lag index (wpli) of input data.
        This implements the method described in:
        Vinck M, Oostenveld R, van Wingerden M, Battaglia F, Pennartz CM.
        "An improved index of phase-synchronization for electrophysiological data 
        in the presence of volume-conduction, noise and sample-size bias."
        Neuroimage. 2011 Apr 15;55(4):1548-65.
        """

        # Check incompatible conditions
        if dojack and isunivariate:
            raise ValueError('Jackknife variance estimates with on-the-fly CSD computation is not supported')

        # On-the-fly computation for univariate data
        if isunivariate:
            if cumtapcnt is None:
                cumtapcnt = np.ones(inputdata.shape[0], dtype=int)

            # Ensure sum of cumtapcnt matches the number of repetitions
            assert np.sum(cumtapcnt) == inputdata.shape[0]

            siz = list(inputdata.shape) + [1]
            nchan = siz[1]
            outsiz = [nchan, nchan] + siz[2:]
            n = len(cumtapcnt)
            sumtapcnt = np.concatenate(([0], np.cumsum(cumtapcnt)))

            # Initialize result arrays
            outsum = np.zeros(outsiz, dtype=complex)
            outsumW = np.zeros(outsiz, dtype=complex)
            outssq = np.zeros(outsiz, dtype=complex)

            for k in range(n):
                indx = np.arange(sumtapcnt[k], sumtapcnt[k + 1])
                for m in np.ndindex(*outsiz[2:]):
                    trial = inputdata[indx].transpose(1, 0, 2)[:, :, m]
                    csdimag = np.imag(trial @ trial.T) / len(indx)
                    outsum[..., m] += csdimag
                    outsumW[..., m] += np.abs(csdimag)
                    outssq[..., m] += (csdimag ** 2)

            if debias:
                wpli = (outsum ** 2 - outssq) / (outsumW ** 2 - outssq)
            else:
                wpli = outsum / outsumW

            v = np.array([])

        else:
            siz = list(inputdata.shape) + [1]
            n = siz[0]

            if n > 1:
                inputdata = np.imag(inputdata)
                outsum = np.nansum(inputdata, axis=0)
                outsumW = np.nansum(np.abs(inputdata), axis=0)
                if debias:
                    outssq = np.nansum(inputdata ** 2, axis=0)
                    wpli = (outsum ** 2 - outssq) / (outsumW ** 2 - outssq)
                else:
                    wpli = outsum / outsumW

                wpli = np.reshape(wpli, siz[1:])

            else:
                wpli = np.full(siz[1:], np.nan)
                print("computation wpli requires >1 trial, returning NaNs")

            # Initialize leave-one-out result arrays
            leave1outsum = np.zeros([1] + siz[1:], dtype=complex)
            leave1outssq = np.zeros([1] + siz[1:], dtype=complex)

            if dojack and n > 2:
                # Compute the variance based on leave-one-out
                for k in range(n):
                    s = outsum - inputdata[k]
                    sw = outsumW - np.abs(inputdata[k])
                    if debias:
                        sq = outssq - inputdata[k] ** 2
                        num = s ** 2 - sq
                        denom = sw ** 2 - sq
                    else:
                        num = s  # estimator of E(Im(X))
                        denom = sw  # estimator of E(|Im(X)|)
                    
                    tmp = num / denom
                    tmp = np.nan_to_num(tmp)
                    leave1outsum += tmp
                    leave1outssq += tmp ** 2

                # Compute the SEM
                n = np.sum(~np.isnan(inputdata), axis=0)
                v = ((n - 1) ** 2 * (leave1outssq - (leave1outsum ** 2 / n)) / (n - 1)).reshape(siz[1:])
                n = np.reshape(n, siz[1:])
            elif dojack:
                v = np.full(siz[1:], np.nan)
            else:
                v = np.array([])

        return wpli, v, n

    def display_connectivity_over_time(self):
        """
        Display the evolution of cortical connectivity over time.
        """
        # Placeholder for the actual connectivity visualization logic
        # This would involve creating a grid structure or using Bokeh for visualization.
        pass  # Replace with actual visualization code

    def calculate_inter_regional_connectivity(self):
        """
        Calculate inter-regional connectivity by averaging wPLI over anatomical groups.
        """
        # Placeholder for the actual inter-regional connectivity calculation logic
        self.inter_regional_connectivity = {
            'Frontal-Parietal': 0.0,  # Replace with actual calculation
            'Prefrontal-Frontal': 0.0  # Replace with actual calculation
        }

    def calculate_frequency_band_wPLI(self):
        """
        Calculate the mean wPLI in defined frequency bands such as delta, theta, and alpha.
        """
        # Placeholder for the actual frequency band wPLI calculation logic
        self.frequency_band_wPLI = {
            'delta': 0.0,  # Replace with actual calculation
            'theta': 0.0,  # Replace with actual calculation
            'alpha': 0.0   # Replace with actual calculation
        }

    def perform_statistical_analysis(self):
        """
        Perform statistical analysis using Linear Mixed Models.
        """
        # Placeholder for the actual statistical analysis logic
        pass  # Replace with actual statistical analysis code

    def calculate_connectivity_dynamics(self):
        """
        Calculate connectivity dynamics including PCA, k-means clustering, and state transitions.
        """
        # Placeholder for the actual connectivity dynamics calculation logic
        self.connectivity_state_index_time_series = {}  # Replace with actual calculation
        self.transition_matrix_counts = {}  # Replace with actual calculation
        self.transition_matrix_probabilities = {}  # Replace with actual calculation

    def exclude_suppression_windows(self, suppression_ratio_threshold=20):
        """
        Exclude time windows with suppression ratio greater than the threshold.

        Parameters:
        suppression_ratio_threshold (int): The suppression ratio threshold.
        """
        # Placeholder for the actual exclusion logic
        pass  # Replace with actual exclusion code

    def aggregate_time_windows(self):
        """
        Aggregate data across all time windows and participants.
        """
        # Placeholder for the actual aggregation logic
        pass  # Replace with actual aggregation code

    def apply_pca(self, n_components):
        """
        Apply PCA for dimensionality reduction.

        Parameters:
        n_components (int): Number of principal components to retain.
        """
        # Placeholder for the actual PCA logic
        pass  # Replace with actual PCA code

    def apply_kmeans(self, n_clusters):
        """
        Apply k-means clustering to classify connectivity patterns.

        Parameters:
        n_clusters (int): Number of clusters.
        """
        # Placeholder for the actual k-means clustering logic
        pass  # Replace with actual k-means clustering code

    def calculate_mean_connectivity_patterns(self):
        """
        Calculate mean connectivity patterns for each state.
        """
        # Placeholder for the actual mean connectivity pattern calculation logic
        pass  # Replace with actual calculation code

    def assign_connectivity_state_index(self):
        """
        Assign a connectivity state index to each cluster.
        """
        # Placeholder for the actual connectivity state index assignment logic
        pass  # Replace with actual assignment code

    def calculate_state_occurrence_and_dwell_time(self):
        """
        Calculate the occurrence rate and dwell time for each state.
        """
        # Placeholder for the actual state occurrence and dwell time calculation logic
        pass  # Replace with actual calculation code

    def construct_transition_matrix(self):
        """
        Construct the transition matrix for the connectivity state time series.
        """
        # Placeholder for the actual transition matrix construction logic
        pass  # Replace with actual matrix construction code

    def exploratory_analysis(self):
        """
        Perform exploratory analysis to determine the most probable states of arrival and departure.
        """
        # Placeholder for the actual exploratory analysis logic
        pass  # Replace with actual exploratory analysis code


"""
# Example usage:
# Assuming eeg_data is an object with EEG data
eeg_data = {'data': 'EEG data'}  # Dummy EEG data object
connectivity_analysis = ConnectivityAnalysis(eeg_data)
connectivity_analysis.calculate_wPLI()
connectivity_analysis.display_connectivity_over_time()
connectivity_analysis.calculate_inter_regional_connectivity()
connectivity_analysis.calculate_frequency_band_wPLI()
connectivity_analysis.perform_statistical_analysis()
connectivity_analysis.calculate_connectivity_dynamics()

print(connectivity_analysis.wpli_matrix)
print(connectivity_analysis.inter_regional_connectivity)
print(connectivity_analysis.frequency_band_wPLI)
print(connectivity_analysis.connectivity_state_index_time_series)
print(connectivity_analysis.transition_matrix_counts)
print(connectivity_analysis.transition_matrix_probabilities)
"""