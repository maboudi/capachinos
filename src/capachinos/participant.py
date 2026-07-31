# Import necessary classes from their respective modules
from .preprocessing.eeg_file import EEGFile
from .preprocessing.eeg_preprocessor import EEGPreprocessor
from .analysis.power_spectral import PowerSpectralAnalysis
from .analysis.connectivity import ConnectivityAnalysis

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
        raise NotImplementedError(
            "A unified CriticalityAnalysis class has not yet been implemented."
        )

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