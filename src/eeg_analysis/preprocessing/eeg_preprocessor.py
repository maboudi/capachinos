import numpy as np
import scipy.signal as signal
import mne
from mne.preprocessing import ICA
from src.eeg_analysis.utils.helpers import create_mne_raw_from_data 

class EEGPreprocessor:
    def __init__(self, eeg_file):
        """
        Initialize the EEGPreprocessor with EEG data from a file.

        Parameters:
        eeg_file: The file containing EEG data, expected to have a participant_id attribute.
        """
        self.participant_id = eeg_file.participant_id
        self.eeg_data = eeg_file.eeg_data
        self.channel_names = eeg_file.channel_names
        self.sampling_frequency = eeg_file.sampling_frequency
        self.ds_sampling_frequency = None
        self.down_sampled_eeg = None
        self.epochs = None
        self.detrended_eeg = None
        self.referenced_eeg = None
        self.filtered_eeg = None
        self.z_scored_eeg = None
        self.segmented_eeg = None
        self.exclude_flags = None
        self.eeg_with_rejected_noisy_segments = None
        self.corrected_eeg = {}
      
    def _check_and_get_attribute(self, attr_name: str):
        """
        Helper method to check for attribute existence and non-None value
        
        Parameters:
        attr_name (str): the name of the attribute to check

        Returns:
        The attribute value if checks pass
        """
        if not hasattr(self, attr_name):
            raise ValueError(f"The attribute {attr_name} does not exist in the EEGPreprocessor class.")
        
        attr_value = getattr(self, attr_name)
        if attr_value is None:
            raise ValueError(f"The attribute {attr_name} is not set in the EEGPreprocessor class.")
        
        return attr_value
    
    def downsample(self, target_fs:int):
        """
        Downsample the EEG data.

        Parameters:
        target_fs (int): The factor by which to downsample the EEG data.
        """
        eeg_data = self.eeg_data
        original_fs = self.sampling_frequency

        decimation_factor = original_fs/target_fs
        if decimation_factor.is_integer():
            decimation_factor = int(decimation_factor)

            # decimate applies an anti-aliasing filter before downsampling
            self.down_sampled_eeg = signal.decimate(eeg_data, q=decimation_factor, axis=0, zero_phase=True)
        else:
            # time consuming, not efficient 

            # it's safer to apply an anti-aliasing filter, even though resample method has it
            # by construction

            signal_duration = eeg_data.shape[0]/original_fs
            num_samples_target = int(signal_duration * target_fs)
            
            num_chan = eeg_data.shape[1]

            nyquist_freq =target_fs/2
            cutoff_freq = nyquist_freq * 0.9
            order = 4
            b,a = signal.butter(order, cutoff_freq/nyquist_freq)

            self.down_sampled_eeg = np.zeros((num_samples_target, num_chan))
            for ch in range(num_chan):
                filtered_signal = signal.filtfilt(b, a, eeg_data[:, ch])
                self.down_sampled_eeg[:, ch] = signal.resample(filtered_signal, num_samples_target)

        self.ds_sampling_frequency = target_fs
        return self.down_sampled_eeg

    def create_epochs(self, events_df, data_attribute_to_process: str = 'eeg_data'):
        """
        Segment the EEG data based on specified events.

        Parameters:
        events_df (DataFrame): A DataFrame where each row contains 'name', 'start_time', and 'end_time' columns
        data_attribute_to_process (str): the name of the attribute to process

        Returns:
        Dictionary with event names as keys and list of arrays of corresponding EEG epochs as values.
        """
        data_to_process = self._check_and_get_attribute(data_attribute_to_process)

        if isinstance(data_to_process, dict):
            raise ValueError(f"The attribute {data_attribute_to_process} is already divided into epochs.")
        else:
            sf = self.ds_sampling_frequency if self.ds_sampling_frequency is not None else self.sampling_frequency

            # Initialize a dictionary to store the epochs
            self.epochs = {event_name: None for event_name in events_df['name'].unique()}
            
            # Iterate over the events, extracting the EEG data for each epoch based on its start and end times
            for _, event in events_df.iterrows():
                start_sample = int(event['start'] * sf)
                end_sample = int(event['end'] * sf)
                epoch = data_to_process[start_sample:end_sample, :]
                self.epochs[event['name']] = epoch

        return self.epochs

    def detrend(self, window_size: float, step_size: float, data_attribute_to_process: str = 'eeg_data'):
        """
        Detrend the EEG data (continuous or epoched) using a local linear regression method.
        
        Parameters:
        window_size (float): the length of each window on which deterneding is applied, in seconds
        step_size (float): the delay between the consequitive windows 
        data_attribute_to_process (str): the name of the attribute to process

        Returns:
        The detrended EEG data, either as a numpy array or a dictionary of epochs.
        """
        data_to_process = self._check_and_get_attribute(data_attribute_to_process)
        sf = self.ds_sampling_frequency if self.ds_sampling_frequency is not None else self.sampling_frequency

        window_size = window_size * sf 
        step_size = step_size * sf 

        # Function to detrend a single epoch or continuous data
        def detrend_epoch(epoch):
            detrended_epoch = np.zeros_like(epoch)
            num_time_points_epoch = epoch.shape[0]
            for start in range(0, num_time_points_epoch, step_size):
                end = min(start + window_size, num_time_points_epoch)
                if end - start < window_size:
                    break
                for channel in range(epoch.shape[1]):
                    segment = epoch[start:end, channel]
                    trend = np.polyfit(range(len(segment)), segment, 1)
                    detrended_epoch[start:end, channel] = segment - (trend[0] * np.arange(len(segment)) + trend[1])
            return detrended_epoch
        
        # Automatically detect if the data is epoched by checking if it's a dictionary
        if isinstance(data_to_process, dict):
            # If data_to_process is a dictionary, detrend each epoch separately
            self.detrended_eeg = {epoch_name: detrend_epoch(epoch_data)
                                 for epoch_name, epoch_data in data_to_process.items()}
        else:
            # If self.eeg_data is not a dictionary, assume it's continuous data and detrend
            self.detrended_eeg = detrend_epoch(data_to_process)

        # Update the self.eeg_data with the detrended result and return it
        return self.detrended_eeg

    def re_reference(self, reference_type: str = 'average'):
        """
        Apply re-referencing to the EEG data (either continuous or epoched).

        Parameters:
        reference_type (str): The type of referencing to apply, default is 'average'.

        Returns:
        The re-referenced EEG data, either as a numpy array or a dictionary of epochs.
        """
        # Function to re-reference a single epoch or continuous data
        def re_reference_epoch(epoch, reference_type):
            if reference_type == 'average':
                # Re-reference the data to the average
                return epoch - np.mean(epoch, axis=0, keepdims=True)
            else:
                raise ValueError("Unsupported reference type")

        # Automatically detect if the data is epoched by checking its type
        if isinstance(self.eeg_data, dict):
            # If self.eeg_data is a dictionary, apply re-referencing to each epoch
            self.referenced_eeg = {epoch_name: re_reference_epoch(epoch_data, reference_type)
                              for epoch_name, epoch_data in self.eeg_data.items()}
        else:
            # If self.eeg_data is not a dictionary, assume it's continuous data
            self.referenced_eeg = re_reference_epoch(self.eeg_data, reference_type)

        # Update the self.eeg_data with the re-referenced result and return it
        return self.referenced_eeg

    def bandpass_filter(self, order: int, low_cutoff: float, high_cutoff: float, data_attribute_to_process: str = 'eeg_data'):
        """
        Apply a bandpass filter to the EEG data.

        Parameters:
        order (int): The order for the bandpass filter
        low_cutoff (float): The low cutoff frequency for the bandpass filter.
        high_cutoff (float): The high cutoff frequency for the bandpass filter.
        data_attribute_to_process (str): the name of the attribute to process
        """
        data_to_process = self._check_and_get_attribute(data_attribute_to_process)
        sf = self.ds_sampling_frequency if self.ds_sampling_frequency is not None else self.sampling_frequency

        nyquist = 0.5 * sf  # Assuming data is already downsampled to 250 Hz
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = signal.butter(order, [low, high], btype='band')

        if isinstance(data_to_process, dict):
            # If data_to_process is a dictionary, bandpass filter each epoch separately
            self.filtered_eeg = {epoch_name: signal.filtfilt(b, a, epoch_data, axis=0)
                                 for epoch_name, epoch_data in data_to_process.items()}
        else:
            # If data_to_process is not a dictionary, assume it's continuous data
            self.filtered_eeg = signal.filtfilt(b, a, data_to_process, axis=0)
        return self.filtered_eeg
    
    def calculate_z_score(self, data_attribute_to_process: str = 'eeg_data'):
        """
        calculate z_scored EEG data (either continuous or epoched)
        data_attribute_to_process (str): the name of the attribute to process
        """
        data_to_process = self._check_and_get_attribute(data_attribute_to_process)

        def calculate_z_score_epoch(epoch):
            z_scored_epoch = np.zeros_like(epoch)
            for channel in range(epoch.shape[1]):
                channel_eeg = epoch[:, channel]
                z_scored_epoch[:, channel] = (channel_eeg - np.mean(channel_eeg))/np.std(channel_eeg)
            return z_scored_epoch
        
        if isinstance(data_to_process, dict):
            self.z_scored_eeg = {epoch_name: calculate_z_score_epoch(epoch_data)
                                 for epoch_name, epoch_data in data_to_process.items()}
        else:
            self.z_scored_eeg = calculate_z_score_epoch(data_to_process)

    def get_segments(self, length: float, data_attribute_to_process: str = 'z_scored_eeg'):
        """
        Divide EEG data (either continuous or epoched) into segments

        Parameters:
        length (float): The length of each segment/window in seconds
        data_attribute_to_process (str): The name of the attribute to process
        """
        data_to_process = self._check_and_get_attribute(data_attribute_to_process)
        segment_length_samples = int(length * self.ds_sampling_frequency)

        def get_segments_epoch(epoch, segment_length_samples):
            # Calculate the number of segments that can be extracted from the epoch
            num_segments = epoch.shape[0] // segment_length_samples

            # Preallocate an array to hold the segments
            segments = np.zeros((segment_length_samples, epoch.shape[1], num_segments))

            # Extract segments
            for i in range(num_segments):
                start_index = i * segment_length_samples
                end_index = start_index + segment_length_samples
                segments[:, :, i] = epoch[start_index:end_index, :]
            
            return segments

        if isinstance(data_to_process, dict):
            self.segmented_eeg = {epoch_name: get_segments_epoch(epoch_data, segment_length_samples)
                                  for epoch_name, epoch_data in data_to_process.items()}
        else:
            self.segmented_eeg = get_segments_epoch(data_to_process, segment_length_samples)
    
    def mark_exclude_segments(self, threshold: float = 2.0, min_channels: int = 4):
        """
        Mark segments that should be excluded based on the amplitude threshold in multiple channels.

        Parameters:
        threshold (float): The amplitude threshold for excluding segments.
        min_channels (int): The minimum number of channels that must exceed the threshold to exclude a segment.
        """
        segmented_eeg = self._check_and_get_attribute('segmented_eeg')
        def mark_exclude_segments_epoch(epoch_data):
            num_segments = epoch_data.shape[2]
            exclude_flags = np.zeros(num_segments, dtype=bool) # True implies exclude, False implies include
            # Extract segments
            for i in range(num_segments):
                segment = epoch_data[:, :, i]
                
                # Get the amplitude for each channel and check against the threshold
                channel_amplitudes = np.mean(np.abs(segment), axis=0)
                # channel_amplitudes = np.max(np.abs(segment), axis=0)
                channels_exceeding = np.sum(channel_amplitudes > threshold)

                if channels_exceeding >= min_channels:
                    exclude_flags[i] = True
            
            return exclude_flags
        
        if isinstance(segmented_eeg, dict):
            self.exclude_flags = {epoch_name: mark_exclude_segments_epoch(epoch_data)
                                  for epoch_name, epoch_data in segmented_eeg.items()}
        else:
            self.exclude_flags = mark_exclude_segments_epoch(segmented_eeg)

    def concatenate_data_excluding_noisy_segments(self, padding = 'nan'):
        """
        Concatenate data after excluding marked segments and replace excluded
        segments with padding or interpolation.

        Parameters:
        padding (str): Strategy for dealing with excluded segments, can be 'zeros', 'nan' or 'interpolate'.
        """
        segmented_eeg = self._check_and_get_attribute('segmented_eeg')
        exclude_flags = self._check_and_get_attribute('exclude_flags')

        def concatenate_data_epoch(epoch_segments, epoch_exclude_flags):
            concatenated_data = []
            for idx, exclude_flag in enumerate(epoch_exclude_flags):
                segment = epoch_segments[:, :, idx]
                if exclude_flag:
                    if padding == 'zeros':
                        concatenated_data.append(np.zeros_like(segment))
                    elif padding == 'nan':
                        concatenated_data.append(np.full_like(segment, np.nan))
                    elif padding == 'interpolate':
                        # Implement an interpolation approach here
                        pass
                    continue  # Skip appending actual segment data if it's excluded

                concatenated_data.append(segment)

            # Combine all segments into a single continuous array
            return np.concatenate(concatenated_data, axis=0)

        if isinstance(segmented_eeg, dict):
            self.eeg_with_rejected_noisy_segments = {epoch_name: concatenate_data_epoch(epoch_segments, exclude_flags[epoch_name])
                                 for epoch_name, epoch_segments in segmented_eeg.items()}
        else:
            self.eeg_with_rejected_noisy_segments = concatenate_data_epoch(segmented_eeg, exclude_flags)
    
    def mark_burst_suppression(self):
        """
        Mark burst suppression periods in the EEG data.
        """
        # Placeholder for actual burst suppression marking logic
        self.marked_eeg = self.eeg_data  # Replace with actual marking operation
        return self.marked_eeg

    def apply_ica(self, epoch_name, data_attribute_to_process: str = 'eeg_with_rejected_noisy_segments'):
        """
        Apply Independent Component Analysis (ICA) to the EEG data.
        """
        data_to_process = self._check_and_get_attribute(data_attribute_to_process)
        data_to_process = data_to_process[epoch_name]

        # Create an MNE Info object with the properties of your data
        ch_names = self.channel_names
        sfreq = self.ds_sampling_frequency if self.ds_sampling_frequency is not None else self.sampling_frequency
        raw = create_mne_raw_from_data(data_to_process, ch_names, sfreq)

        # Fit ICA
        ica = ICA(n_components=0.95, random_state=97)
        ica.fit(raw)

        # Depending on your artifact, you might have prior knowledge on which components to exclude.
        # Assuming you have the components to exclude from some external information (e.g., a list of indices)
        ica.exclude = [0, 1]  # Put the indices of components associated with eye blinks

        # Apply ICA to the Raw object to remove the components
        raw_corrected = ica.apply(raw.copy())

        self.corrected_eeg[epoch_name] = raw_corrected.get_data().T

    def process(self, operations):
        """
        Process the EEG data following a specified sequence of operations with arguments.

        Parameters:
        operations (list of tuples):  A list where each element is a tuple containing the 
                                     operation name as a string and a dictionary of arguments.
        """
        operations_map = {
            'downsample': self.downsample,
            'create_epochs': self.create_epochs,
            'detrend': self.detrend, 
            're_reference': self.re_reference,
            'bandpass_filter': self.bandpass_filter,
            'calculate_z_score': self.calculate_z_score,
            'get_segments': self.get_segments,
            'mark_exclude_segments':self.mark_exclude_segments,
            'concatenate_data_excluding_noisy_segments': self.concatenate_data_excluding_noisy_segments
        }

        for op_name, args in operations:
            if op_name in operations_map:
                operation_method = operations_map[op_name]
                result = operation_method(**args)
                if result is not None:
                    self.eeg_data = result
            else:
                raise ValueError("Unsupported operation: {}".format(op_name))