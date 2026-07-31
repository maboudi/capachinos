import numpy as np
import pandas as pd

class EEGFile:
    def __init__(self, participant_id, vhdr_path, vmrk_path, eeg_path):
        """
        Initialize the EEGFile with paths to the BrainVision EEG files.

        Parameters:
        participant_id: The ID of the participant.
        vhdr_path: Path to the .vhdr file.
        vmrk_path: Path to the .vmrk file.
        eeg_path: Path to the .eeg file.
        """
        self.participant_id = participant_id
        self.vhdr_path = vhdr_path
        self.vmrk_path = vmrk_path
        self.eeg_path = eeg_path
        self.metadata = {}
        self.channel_names = []
        self.markers_df = []
        self.events_df = []
        self.eeg_data = None
        self.sampling_interval = None
        self.sampling_frequency = None
        self.channel_groups = {
            'prefrontal': ['Fp1', 'Fp2', 'AFz'],
            'frontal': ['F5', 'F6', 'Fz'],
            'central': ['C3', 'C4', 'Cz'],
            'temporal': ['T7', 'T8'],
            'parietal': ['P5', 'P6', 'Pz'],
            'occipital': ['O1', 'O2'],
        }

    def read_vhdr(self):
        """
        Read and parse the .vhdr file to extract metadata.
        """
        try:
            with open(self.vhdr_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    if key.startswith('ch'):
                        parts = value.strip().split(',')
                        self.metadata[key.strip()] = parts[0].strip()
                        self.channel_names.append(parts[0].strip())
                    else:
                        self.metadata[key.strip()] = value.strip()
        except Exception as e:
            print(f"Error reading .vhdr file: {e}")

    def read_vmrk(self):
        """
        Read and parse the .vmrk file to extract markers.
        """
        markers = {
            '1a': 'pre-op rest start',
            '1b': 'pre-op rest end',
            '2': 'loss of consciousness',
            '3': 'intubation',
            '4': 'skin incision',
            '5': 'drug infusion',  # start
            '6': 'extubation',
            '7a': 'pacu rest start',
            '7b': 'pacu rest end'
        }
        
        # Initialize an empty list to store marker information
        marker_list = []
        seen_indices = {'1': 0, '7': 0}

        try:
            with open(self.vmrk_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                if line.startswith('Mk'):
                    parts = line.strip().split(',')

                    if parts[1] in ['1', '7']:
                        count = seen_indices[parts[1]]
                        marker_index = parts[1] + ('a' if count == 0 else 'b')  # Choose 'a' or 'b'
                        seen_indices[parts[1]] = count + 1  # Increment the count
                    else:
                        marker_index = parts[1]   
                        
                    if marker_index in markers:
                        marker_info = {
                                'marker_index': marker_index,
                                'time': float(parts[2])/self.sampling_frequency,
                                'description': markers[marker_index]
                            }
                        marker_list.append(marker_info)
            
            # Convert the list of dictionaries to a pandas DataFrame
            self.markers_df = pd.DataFrame(marker_list)

        except Exception as e:
            print(f"Error reading .vmrk file: {e}")

        # Use the helper function to define events, explicitly specifying start or end markers or a duration
        event_list = [
            self.create_event('pre_preop_rest', end_marker='pre-op rest start', duration=1200),
            self.create_event('preop_rest', start_marker='pre-op rest start', end_marker='pre-op rest end'),
            self.create_event('loc', end_marker='loss of consciousness', duration=300),
            self.create_event('pre_incision', end_marker='skin incision', duration= 300),
            self.create_event('maintenance', start_marker='loss of consciousness', end_marker='drug infusion'),
            self.create_event('pre_drug_infusion', end_marker='drug infusion', duration=3000),
            self.create_event('emergence', start_marker='drug infusion', end_marker='extubation'), 
            self.create_event('pacu_rest', start_marker='pacu rest start', end_marker='pacu rest end')
        ]

        self.events_df = pd.DataFrame(event_list)
        self.events_df['duration'] = self.events_df['end'] - self.events_df['start']


    def create_event(self, name, start_marker=None, end_marker=None, duration=None):
        """
        Creates an event dictionary using either a specific start and end marker,
        or a start or end marker with a fixed duration.
        
        Parameters:
        - name: name of the event
        - start_marker: marker description for the start time of the event, if available
        - end_marker: marker description for the end time of the event, if available
        - duration: duration in seconds to add to the start time, if no end marker is provided

        Returns a dictionary representing the event with 'name', 'start', and 'end' times.
        """
        total_duration = int(self.eeg_data.shape[0]/self.sampling_frequency) # in second

        start_time = None
        end_time = None

        if start_marker and start_marker in self.markers_df['description'].values:
            start_time = self.markers_df.loc[self.markers_df['description'] == start_marker, 'time'].values[0]

        if end_marker and end_marker in self.markers_df['description'].values:
            end_time = self.markers_df.loc[self.markers_df['description'] == end_marker, 'time'].values[0]

        # If no end marker is provided, use the start marker and duration to define the event
        if start_time is not None and end_time is None and duration is not None:
            end_time = min(total_duration, start_time + duration)

        # If no start marker is provided, use the end marker and duration to define the event
        if start_time is None and end_time is not None and duration is not None:
            start_time = max(0, end_time - duration)

        # Error checking to ensure we have valid start and end times
        if start_time is None or end_time is None:
            raise ValueError(f"Event '{name}' could not be created due to missing markers or invalid duration.")

        return {'name': name, 'start': start_time, 'end': end_time}


    def read_eeg(self):
        """
        Read and parse the .eeg file to extract EEG data.
        """
        try:
            eeg_data_format = self.metadata.get('DataFormat', 'BINARY')
            eeg_data_type = np.float32 if 'IEEE_FLOAT_32' in self.metadata.get('BinaryFormat', '') else np.int16
            num_channels = int(self.metadata.get('NumberOfChannels', 0))
            sampling_interval = float(self.metadata.get('SamplingInterval', 0)) / 1e6  # convert to seconds

            self.sampling_interval = sampling_interval
            self.sampling_frequency = int(1/sampling_interval)
            
            with open(self.eeg_path, 'rb') as file:
                if eeg_data_format == 'BINARY':
                    raw_data = np.fromfile(file, dtype=eeg_data_type)
                    self.eeg_data = raw_data.reshape(-1, num_channels)

            print("EEG Data Shape:", self.eeg_data.shape)  # Add this line to verify data shape

        except Exception as e:
            print(f"Error reading .eeg file: {e}")

    def load_data(self):
        """
        Load and parse all EEG files (.vhdr, .vmrk, .eeg).
        """
        self.read_vhdr()
        self.read_eeg()
        self.read_vmrk()