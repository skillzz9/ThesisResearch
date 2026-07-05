import numpy as np
from .spectral_analyzer import SpectralAnalyzer
from .temporal_analyzer import TemporalAnalyzer

class RightEyePipeline:
    """
    Orchestrates the Right Eye branch: Spectral Analytics + Temporal Analytics.
    """
    def __init__(self):
        self.spectral_analyzer = SpectralAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()

    def get_vector(self, file_path):
        """
        Processes an audio file and returns the combined flat 1D array of fixed length.
        """
        spectral_features = self.spectral_analyzer.extract(file_path)
        temporal_features = self.temporal_analyzer.extract(file_path)
        
        # Assemble into a single, flat 1D array
        return np.concatenate([spectral_features, temporal_features])
