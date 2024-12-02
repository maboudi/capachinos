import numpy as np
import scipy.signal as signal
import mne
from mne.preprocessing import ICA
from src.eeg_analysis.utils.helpers import create_mne_raw_from_data
from src.eeg_analysis.utils.helpers import calculate_z_score_eeg

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
        self.channel_groups = eeg_file.channel_groups
        self.sampling_frequency = eeg_file.sampling_frequency
        self.down_sampled_eeg = None
        self.epochs = None
        self.detrended_eeg = None
        self.referenced_eeg = None
        self.filtered_eeg = None
        self.eeg_with_rejected_noisy_segments = None
        self.ica_corrected_eeg = {}
      
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

        self.sampling_frequency = target_fs

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
            sf = self.sampling_frequency

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
        sf = self.sampling_frequency

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

        return self.detrended_eeg

    def re_reference(self, reference_type: str = 'average', data_attribute_to_process='eeg_data'):
        """
        Apply re-referencing to the EEG data (either continuous or epoched).

        Parameters:
        reference_type (str): The type of referencing to apply, default is 'average'.

        Returns:
        The re-referenced EEG data, either as a numpy array or a dictionary of epochs.
        """
        data_to_process = self._check_and_get_attribute(data_attribute_to_process)

        # Function to re-reference a single epoch or continuous data
        def re_reference_epoch(epoch, reference_type):
            if reference_type == 'average':
                # Re-reference the data to the average
                return epoch - np.mean(epoch, axis=0, keepdims=True)
            else:
                raise ValueError("Unsupported reference type")

        # Automatically detect if the data is epoched by checking its type
        if isinstance(data_to_process, dict):
            # If self.eeg_data is a dictionary, apply re-referencing to each epoch
            self.referenced_eeg = {epoch_name: re_reference_epoch(epoch_data, reference_type)
                              for epoch_name, epoch_data in data_to_process.items()}
        else:
            # If self.eeg_data is not a dictionary, assume it's continuous data
            self.referenced_eeg = re_reference_epoch(data_to_process, reference_type)

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
        sf = self.sampling_frequency

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
    
    def exclude_noisy_periods(self, window_size=None, threshold: float = 2.0, min_num_channels: int = 4, padding = 'nan'):
        """
        Identify and exclude noisy periods from the EEG data using a z-score threshold approach.
        This method divides the EEG data into windows, calculates the z-score for each window,
        and then excludes any segments that exceed the threshold in a sufficient number of channels.
        Excluded segments can be zeroed out, filled with NaN, or interpolated.

        Parameters:
        window_size (float, optional): The length of each window in seconds for segmenting the EEG data.
                                    If None, a default size of 2.0 seconds is used.
        threshold (float, optional): The z-score threshold above which a window is considered noisy.
                                    Default value is 2.0, representing typically around the 95th percentile.
        min_num_channels (int, optional): The minimum number of channels that need to exceed the
                                        threshold for a segment to be marked for exclusion.
                                        Default value is 4.
        padding (str, optional): The method of handling the excluded segments, with options being:
                                'zeros' to fill with zeros, 'nan' to fill with NaNs, or 'interpolate' to use interpolation.
                                Default is 'nan'.
        
        Returns:
        np.ndarray or dict: The EEG data with noisy segments handled according to the padding parameter.
                            Returns a dictionary if the EEG data was initially epoched, or a NumPy array for continuous data.
        """
        eeg_data = self.eeg_data
        sampling_frequency = self.sampling_frequency

        if window_size is None:
            window_size = 2.0

        z_scored_eeg = self.calculate_z_score(eeg_data, self.sampling_frequency) # or can be called EEGPreprocessor.calculate_z_score 
        
        segmented_eeg, _ = self.get_segments(eeg_data, sampling_frequency, window_size=window_size)
        segmented_z_scored_eeg, _ = self.get_segments(z_scored_eeg, sampling_frequency, window_size=window_size)

        exclude_indices = self._mark_exclude_segments(segmented_z_scored_eeg, threshold=threshold, min_num_channels=min_num_channels)

        def concatenate_data_epoch(epoch_segments, epoch_exclude_indices):
            concatenated_data = []
            for idx, exclude_flag in enumerate(epoch_exclude_indices):
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
            return np.concatenate(concatenated_data, axis=0)

        if isinstance(segmented_eeg, dict):
            self.eeg_with_rejected_noisy_segments = {epoch_name: concatenate_data_epoch(epoch_segments, exclude_indices[epoch_name])
                                 for epoch_name, epoch_segments in segmented_eeg.items()}
        else:
            self.eeg_with_rejected_noisy_segments = concatenate_data_epoch(segmented_eeg, exclude_indices)

        return self.eeg_with_rejected_noisy_segments
    
    @staticmethod
    def calculate_z_score(data, sampling_frequency):
        """
        Calculate the z-score normalized version of the provided EEG data. Z-scoring is 
        performed channel-wise. This static method supports continuous data, epoched data, 
        segmented data, or data that is both epoched and segmented.

        Parameters:
        data (np.ndarray or dict): Input data which can be a NumPy array or a dictionary of NumPy arrays.
                                The input NumPy array can be 2D (time x channels) for continuous data,
                                or 3D (time x channels x segments) for segmented data.
                                If a dictionary is provided, each value should be a NumPy array of epoched data.
        
        Returns:
        np.ndarray or dict: The z-score normalized EEG data in the same structure as the input.
                            Each channel within the epochs or segments will be independently normalized.
        """
        def calculate_z_score_epoch(epoch):
            # return calculate_z_score_eeg(
            #     epoch,
            #     duration=120,
            #     sampling_rate=sampling_frequency,
            #     peak_threshold=8)
            z_scored_epoch = np.zeros_like(epoch)
            for channel in range(epoch.shape[1]):
                channel_eeg = epoch[:, channel]
                z_scored_epoch[:, channel] = (channel_eeg - np.mean(channel_eeg))/np.std(channel_eeg)
            return z_scored_epoch
        
        def calculate_z_score_segmented(segmented_data):
            z_score_segmented = np.zeros_like(segmented_data)
            for seg_index in range(segmented_data.shape[2]):
                z_score_segmented[:, :, seg_index] = calculate_z_score_epoch(segmented_data[:, :, seg_index])
            return z_score_segmented

        if isinstance(data, dict):
            first_key = next(iter(data)) # get the first key to examine the structure
            if isinstance(data[first_key], np.ndarray) and data[first_key].ndim == 3:
                # Dictionary of 3D arrays, epoched and segmented data
                z_scored_eeg = {
                    epoch_name: calculate_z_score_segmented(epoch_data)
                    for epoch_name, epoch_data in data.items()
                }
            else:
                # Dictionary of 2D arrays, epoched data but not segmented
                z_scored_eeg = {
                    epoch_name: calculate_z_score_epoch(epoch_data)
                    for epoch_name, epoch_data in data.items()
                }
        elif data.ndim ==3:
            z_scored_eeg = calculate_z_score_segmented(data)
        else:
            z_scored_eeg = calculate_z_score_epoch(data)

        return z_scored_eeg

    @staticmethod
    def get_segments(data, sampling_frequency, window_size: float, overlap: float = None):
        """
        Divide EEG data (either continuous or epoched) into (overlapping) segments and return window centers.

        Parameters:
        data: can be continuous or epoched data
        sampling_frequency (float): sampling frequency of EEG data
        window_size (float): The length of each segment/window in seconds
        overlap (float): The percentage overlap between consecutive segments (0 <= overlap < 1)
        
        Returns:
        segmented_eeg: Segmented EEG data, either in a dictionary if input is a dictionary or a numpy array otherwise
        window_centers: List of window center indices in seconds
        """
        if overlap is None:
            overlap = 0.0

        if not (0 <= overlap < 1):
            raise ValueError("Overlap must be between 0 (inclusive) and 1 (non-inclusive)")

        segment_length_samples = int(window_size * sampling_frequency)
        step_size_samples = int(segment_length_samples * (1 - overlap))

        def get_segments_epoch(epoch, segment_length_samples, step_size_samples):
            epoch_length_samples = epoch.shape[0]
            
            # If the epoch length is smaller than the segment length, handle it by creating a single segment
            if epoch_length_samples < segment_length_samples:
                segments = np.zeros((segment_length_samples, epoch.shape[1], 1))
                segments[:epoch_length_samples, :, 0] = epoch
                window_centers = [epoch_length_samples / 2 / sampling_frequency]
                return segments, window_centers

            # Calculate the number of segments for longer data
            num_segments = (epoch_length_samples - segment_length_samples) // step_size_samples + 1
            segments = np.zeros((segment_length_samples, epoch.shape[1], num_segments))
            window_centers = []

            for i in range(num_segments):
                start_index = i * step_size_samples
                end_index = start_index + segment_length_samples
                segments[:, :, i] = epoch[start_index:end_index, :]
                window_centers.append((start_index + end_index) / 2 / sampling_frequency)

            return segments, window_centers

        if isinstance(data, dict):
            segmented_eeg = {}
            window_centers = {}
            for epoch_name, epoch_data in data.items():
                segments, centers = get_segments_epoch(epoch_data, segment_length_samples, step_size_samples)
                segmented_eeg[epoch_name] = segments
                window_centers[epoch_name] = centers
        else:
            segmented_eeg, window_centers = get_segments_epoch(data, segment_length_samples, step_size_samples)

        return segmented_eeg, window_centers
    
    @staticmethod
    def _mark_exclude_segments(segmented_eeg, threshold: float = 2.0, min_num_channels: int = 4):
        """
        Mark segments that should be excluded based on the amplitude threshold in multiple channels.

        Parameters:
        threshold (float): The amplitude threshold for excluding segments.
        min_channels (int): The minimum number of channels that must exceed the threshold to exclude a segment.
        """
        def mark_exclude_segments_epoch(epoch_data):
            num_segments = epoch_data.shape[2]
            exclude_indices = np.zeros(num_segments, dtype=bool) # True implies exclude, False implies include
            # Extract segments
            for i in range(num_segments):
                segment = epoch_data[:, :, i]
                
                # Get the amplitude for each channel and check against the threshold
                # channel_amplitudes = np.mean(np.abs(segment), axis=0)
                channel_amplitudes = np.max(np.abs(segment), axis=0)
                channels_exceeding = np.sum(channel_amplitudes > threshold)

                if channels_exceeding >= min_num_channels:
                    exclude_indices[i] = True
            
            return exclude_indices
        
        if isinstance(segmented_eeg, dict):
            exclude_indices = {epoch_name: mark_exclude_segments_epoch(epoch_data)
                                  for epoch_name, epoch_data in segmented_eeg.items()}
        else:
            exclude_indices = mark_exclude_segments_epoch(segmented_eeg)

        return exclude_indices

    def apply_ica(self, epoch_name, data_attribute_to_process: str = 'eeg_with_rejected_noisy_segments'):
        """
        Apply Independent Component Analysis (ICA) to the EEG data.
        """
        data_to_process = self._check_and_get_attribute(data_attribute_to_process)
        data_to_process = data_to_process[epoch_name]

        # Create an MNE Info object with the properties of your data
        ch_names = self.channel_names
        sf = self.sampling_frequency
        raw = create_mne_raw_from_data(data_to_process, ch_names, sf)

        # Fit ICA
        ica = ICA(n_components=0.95, random_state=97)
        ica.fit(raw)

        # Depending on your artifact, you might have prior knowledge on which components to exclude.
        # Assuming you have the components to exclude from some external information (e.g., a list of indices)
        ica.exclude = [0]  # Put the indices of components associated with eye blinks

        # Apply ICA to the Raw object to remove the components
        raw_corrected = ica.apply(raw.copy())

        self.ica_corrected_eeg[epoch_name] = raw_corrected.get_data().T

        return self.ica_corrected_eeg

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
            'exclude_noisy_periods': self.exclude_noisy_periods
        }

        for op_name, args in operations:
            if op_name in operations_map:
                result = operations_map[op_name](**args)

                if result is not None:
                    self.eeg_data = result
            else:
                raise ValueError("Unsupported operation: {}".format(op_name))
            
    def convert_to_neuroscope_eeg(eeg_data, sampling_rate, eeg_path, suffix):
        
        def convert_to_int(eeg_data):
            num_channels = eeg_data.shape[1]
            num_samples = eeg_data.shape[0]
            eeg_data_converted = np.zeros_like(eeg_data, dtype=np.int16)

            # Calculate the number of samples corresponding to 20 seconds
            bout_samples = sampling_rate * 20
            
            # Process each channel
            for i in range(num_channels):
                channel_data = eeg_data[:, i]
                
                # Process in bouts of 20 seconds
                for start in range(0, num_samples, bout_samples):
                    end = min(start + bout_samples, num_samples)
                    segment = channel_data[start:end]
                    
                    # Calculate mean and standard deviation
                    segment_mean = np.mean(segment)
                    segment_std = np.std(segment)
                    
                    # Define the desired range: [-10 SD, +10 SD]
                    lower_bound = segment_mean - 5 * segment_std
                    upper_bound = segment_mean + 5 * segment_std
                    
                    # Normalize and scale to int16 range
                    if upper_bound != lower_bound:
                        segment_normalized = (segment - lower_bound) / (upper_bound - lower_bound)
                        eeg_data_converted[start:end, i] = (
                            (segment_normalized * (np.iinfo(np.int16).max - np.iinfo(np.int16).min)) + 
                            np.iinfo(np.int16).min
                        ).astype(np.int16)
                    else:
                        # Handle the case where upper_bound equals lower_bound (unlikely but possible)
                        eeg_data_converted[start:end, i] = 0
                
            return eeg_data_converted

        # Save the converted data
        def save_converted_data(eeg_data, output_path):
            # eeg_data.tofile(output_path)            
            with open(output_path, 'wb') as file:
                file.write(eeg_data.tobytes())

        import os
        def generate_new_file_path(eeg_path, suffix):
            base_name, extension = os.path.splitext(eeg_path)
            new_base_name = base_name + suffix
            new_file_path = new_base_name + extension
            return new_file_path

        # Paths
        output_eeg_path = generate_new_file_path(eeg_path, suffix)
        
        eeg_data_int = convert_to_int(eeg_data)
        eeg_data_int_transposed = eeg_data_int
        # print("EEG Data Shape (after conversion to int16):", eeg_data_int.shape)  # Verification

        save_converted_data(eeg_data_int_transposed, output_eeg_path)

        return eeg_data_int_transposed
