import collections
import collections.abc
# Monkey-patch for madmom compatibility with Python 3.10+
for name in ['MutableSequence', 'MutableMapping', 'Iterable', 'Mapping', 'Sequence', 'Callable']:
    if not hasattr(collections, name) and hasattr(collections.abc, name):
        setattr(collections, name, getattr(collections.abc, name))

import numpy as np
# Monkey-patch for madmom compatibility with NumPy 1.24+
np.float = np.float64
np.int = np.int64
np.bool = np.bool_
np.complex = np.complex128
np.object = object
np.str = str

from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor

class TemporalAnalyzer:
    """Handles temporal analytics (BPM, beat confidence) via madmom."""
    def __init__(self):
        self.beat_processor = RNNBeatProcessor()
        # TempoEstimator was a typo, the actual class is TempoEstimationProcessor
        self.tempo_estimator = TempoEstimationProcessor(fps=100)

    def extract(self, file_path):
        """Extracts BPM and beat confidence."""
        act = self.beat_processor(file_path)
        tempos = self.tempo_estimator(act)
        
        if len(tempos) > 0:
            bpm = tempos[0][0]
            confidence = tempos[0][1]
        else:
            bpm = 0.0
            confidence = 0.0
            
        return np.array([bpm, confidence])
