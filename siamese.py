import torch
import torch.nn as nn
from towers import MusicTower

class SiameseNetwork(nn.Module):
    """
    True Siamese Network wrapper for Metric Learning.
    Contains ONE instance of MusicTower. 
    Because both Loop A and Loop B are passed through the exact same object,
    their weights are 100% shared, guaranteeing a perfectly symmetrical embedding space.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Instantiate a SINGLE tower (Shared Weights)
        self.tower = MusicTower()
        
    def forward(self, image_A, math_A, image_B, math_B, image_C=None, math_C=None):
        """
        Forward pass for loops simultaneously through the shared network.
        Returns the unit hypersphere embeddings.
        """
        embed_A = self.tower(image_A, math_A)
        embed_B = self.tower(image_B, math_B)
        
        if image_C is not None and math_C is not None:
            embed_C = self.tower(image_C, math_C)
            return embed_A, embed_B, embed_C
            
        return embed_A, embed_B
