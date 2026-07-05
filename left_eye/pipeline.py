import torch
from .audio_loader import AudioLoader
from .feature_extractor import VisualFeatureExtractor
from .cnn_backbone import ResNet18FeatureExtractor

class LeftEyePipeline:
    """
    Orchestrates the Left Eye branch: Audio Loading -> Feature Extraction -> CNN Backbone.
    """
    def __init__(self, sample_rate=22050):
        self.audio_loader = AudioLoader(sample_rate=sample_rate)
        self.feature_extractor = VisualFeatureExtractor(sample_rate=sample_rate)
        self.cnn = ResNet18FeatureExtractor()

    def get_vector_from_array(self, y, sr=22050):
        """
        Processes an audio array directly and returns the (3, 84, Time) tensor.
        """
        stacked_features = self.feature_extractor.extract_and_stack(y)
        tensor_input = torch.tensor(stacked_features, dtype=torch.float32)
        return tensor_input

    def get_vector(self, file_path):
        
        # 4. Pass through CNN
        return self.cnn.forward(tensor_input)
