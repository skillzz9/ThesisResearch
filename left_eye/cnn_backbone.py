import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18FeatureExtractor:
    """Modified ResNet-18 for extracting a fixed 1D vector from the visual features."""
    def __init__(self):
        try:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            self.model = models.resnet18(pretrained=True)
            
        # Remove the final fully connected layer by replacing it with Identity.
        self.model.fc = nn.Identity()
        
        # Replace the Average Pooling layer with Max Pooling so it only captures the loudest/strongest notes
        # and ignores the silent (zero-activation) parts of the loop.
        self.model.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Set to evaluation mode
        self.model.eval()

    def forward(self, tensor_input):
        """Passes the 3-channel tensor through the network to get a flattened 512-D vector."""
        with torch.no_grad():
            output = self.model(tensor_input)
            
        # Flatten the tensor to ensure it is exactly 512 features
        return output.flatten()
