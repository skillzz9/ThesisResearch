import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Standard Contrastive Loss for Metric Learning.
    Penalizes distance for compatible pairs (label=1) and 
    penalizes closeness for incompatible pairs (label=0) within a margin.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embed_A, embed_B, label):
        """
        embed_A, embed_B: L2 normalized embeddings of shape [Batch, 64]
        label: 1 for compatible (similar), 0 for incompatible (dissimilar)
        """
        # Calculate Euclidean distance between embeddings
        # keepdim=True ensures shape [Batch, 1] for reliable broadcasting
        euclidean_distance = F.pairwise_distance(embed_A, embed_B, keepdim=True)
        
        # Loss for compatible pairs (label == 1): Distance squared
        loss_compatible = label * torch.pow(euclidean_distance, 2)
        
        # Loss for incompatible pairs (label == 0): (Margin - Distance) squared
        # Clamp ensures we only penalize if the distance is less than the margin
        loss_incompatible = (1.0 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        # Total contrastive loss is the mean over the batch
        loss = torch.mean(loss_compatible + loss_incompatible)
        
        return loss
