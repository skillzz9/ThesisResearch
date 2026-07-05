import librosa

class AudioLoader:
    """Loads audio files for the visual pipeline."""
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate

    def load(self, file_path):
        """Loads audio file at a standard sample rate using librosa."""
        y, sr = librosa.load(file_path, sr=self.sr)
        return y, sr
