import librosa
import numpy as np

class TemporalAnalyzer:
    """Handles temporal analytics (BPM) via librosa (bypassing madmom for Python 3.12 compatibility)."""
    def __init__(self):
        pass

    def extract(self, file_path):
        """Extracts BPM using librosa instead of madmom."""
        y, sr = librosa.load(file_path, sr=22050)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # In newer librosa versions, tempo is returned as a 1D array
        if isinstance(tempo, np.ndarray):
            bpm = tempo[0]
        else:
            bpm = tempo
            
        # Hardcode confidence to 1.0 since librosa doesn't output confidence
        confidence = 1.0
            
        return np.array([bpm, confidence])
