# Import necessary classes from their respective modules
from preprocessing import EEGFile
from preprocessing import EEGPreprocessor
from analysis import PowerSpectralAnalysis
from analysis import ConnectivityAnalysis
from analysis import CriticalityAnalysis

class ParticipantData:
    def __init__(self, participant_id):
        self.participant_id = participant_id
        self.experiment = None
        self.raw_eeg_file = None
        self.preprocessed_eeg = None
        self.power_spectral_analysis = None
        self.connectivity_analysis = None
        self.criticality_analysis = None
        
    def load_raw_data(self, file_path):
        self.raw_eeg_file = EEGFile(self.participant_id, file_path)
        self.raw_eeg_file.load_data()
        
    def preprocess_data(self):
        # Assuming preprocess method exists and prepares the data for analysis
        self.preprocessed_eeg = EEGPreprocessor(self.raw_eeg_file).preprocess()
        
    def analyze_power_spectral(self):
        # Assuming an appropriate method exists in the PowerSpectralAnalysis class
        self.power_spectral_analysis = PowerSpectralAnalysis(self.preprocessed_eeg).analyze()
        
    def analyze_connectivity(self):
        # Assuming an appropriate method exists in the ConnectivityAnalysis class
        self.connectivity_analysis = ConnectivityAnalysis(self.preprocessed_eeg).analyze()
        
    def analyze_criticality(self):
        # Assuming an appropriate method exists in the CriticalityAnalysis class
        self.criticality_analysis = CriticalityAnalysis(self.preprocessed_eeg).analyze()

"""
# Usage Example
participant_id = 'P001'
file_path = 'path_to_eeg_file_for_P001'
participant_data = ParticipantData(participant_id)

# Load, preprocess, and perform analyses
participant_data.load_raw_data(file_path)
participant_data.preprocess_data()
participant_data.analyze_power_spectral()
participant_data.analyze_connectivity()
participant_data.analyze_criticality()

# Now participant_data holds all the results from the different analyses.
"""