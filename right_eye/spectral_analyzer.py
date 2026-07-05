import opensmile
import librosa
import numpy as np

class SpectralAnalyzer:
    """Handles rigid spectral analytics via openSMILE."""
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract(self, file_path):
        """Extracts statistical averages for the entire loop, explicitly dropping silence."""
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Use an energy mask to split the audio into non-silent intervals
        # top_db=40 drops anything that is 40dB quieter than the peak loudness
        intervals = librosa.effects.split(y, top_db=40)
        
        if len(intervals) == 0:
            # If everything was silence (which shouldn't happen with our loop script, but as a safety net)
            return np.zeros(88) # eGeMAPS has exactly 88 features
            
        # Concatenate only the parts of the audio that actually contain sound
        y_non_silent = np.concatenate([y[start:end] for start, end in intervals])
        
        # Process the newly stitched non-silent signal directly
        df = self.smile.process_signal(y_non_silent, sr)
        return df.iloc[0].values
