import librosa
import numpy as np
import scipy.ndimage

class VisualFeatureExtractor:
    """Extracts and spatially aligns visual features (CQT & Chromagram) with silence masking."""
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate

    def extract_and_stack(self, y):
        """
        Extracts CQT and Chromagram, applies Silence Weighting (RMS masking),
        aligns them spatially, and stacks them into a 3-channel tensor: [CQT, Chromagram, CQT].
        """
        # 1. Extract Raw Features
        # Constant-Q Transform (CQT) (84 vertical bins), then LOG-COMPRESS to dB and
        # normalise to [0,1]. Linear magnitude buries quiet notes/harmonics near zero;
        # dB scaling makes the full dynamic range visible to the CNN. Using the clip's
        # own max as reference also removes absolute loudness (an instrument
        # fingerprint the model should not use).
        cqt = np.abs(librosa.cqt(y, sr=self.sr, n_bins=84))
        if cqt.max() > 0:
            cqt = librosa.amplitude_to_db(cqt, ref=np.max, top_db=80.0)   # [-80, 0]
            cqt = (cqt + 80.0) / 80.0                                     # [0, 1]

        # Chromagram (12 vertical bins; per-frame normalised by librosa)
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_chroma=12)
        
        # 2. Silence Weighting via RMS Energy Masking
        # Calculate frame-by-frame Root Mean Square (RMS) energy.
        # librosa.feature.rms returns shape (1, t), so we take [0] to get a 1D array of length t.
        rms = librosa.feature.rms(y=y)[0]
        
        # Define a dynamic threshold: 10% of max RMS energy, but strictly no less than 1e-3.
        threshold = max(1e-3, 0.1 * np.max(rms))
        
        # Create a 1D float mask (1.0 for active frames above threshold, 0.0 for silence)
        mask = (rms >= threshold).astype(np.float32)
        
        # Apply the Mask: Zero out the vertical columns (time frames) that fall below the threshold.
        # Broadcasting naturally multiplies the (t,) mask against the (84, t) and (12, t) matrices.
        cqt = cqt * mask
        chroma = chroma * mask
        
        # 3. Spatial Alignment: Stretch Chromagram from 12 to 84 bins
        zoom_factor = (84 / 12, 1)
        chroma_stretched = scipy.ndimage.zoom(chroma, zoom_factor, order=0)
        
        # 4. Tensor Stacking: Create a 3-channel tensor
        stacked_features = np.stack([cqt, chroma_stretched, cqt], axis=0)
        
        return stacked_features
