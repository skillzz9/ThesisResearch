import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MusicTower(nn.Module):
    """
    Twin architecture processing an image_tensor and a math_vector.
    Outputs an L2-normalized 64-dimensional embedding.
    """
    def __init__(self):
        super(MusicTower, self).__init__()
        
        # --- Visual Branch (Left Eye) ---
        # Load pre-trained ResNet-18
        try:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            self.resnet = models.resnet18(pretrained=True)
            
        # 1. ARCHITECTURE UPGRADE: Unfreeze the ENTIRE ResNet!
        # By removing the freeze lock on conv1 and layer1, the network will discard its
        # pre-trained "dog/car edge detectors" and learn to detect audio frequency harmonics natively!
        for name, child in self.resnet.named_children():
            for param in child.parameters():
                param.requires_grad = True
                    
        self.visual_dropout = nn.Dropout(p=0.2)
                    
        # Replace the Average Pooling layer with Adaptive Max Pooling
        self.resnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Remove the final fully connected layer by replacing it with Identity.
        # (ResNet forward pass flattens before passing to fc, so Identity leaves it as a flattened 512-dim vector)
        self.resnet.fc = nn.Identity()
        
        # --- Mathematical Branch (Right Eye) ---
        # MLP for 90-dim math_vector
        self.math_mlp = nn.Sequential(
            nn.BatchNorm1d(90), # FIX: Dynamically scales the raw [BPM, flux, etc.] on the GPU!
            nn.Linear(90, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.3)
        )
        
        # --- Fusion and Output Space ---
        # 512 (Visual) + 128 (Math) = 640
        self.fusion_mlp = nn.Sequential(
            nn.Linear(640, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 64)
        )
        
    def forward(self, image_tensor, math_vector):
        """
        image_tensor: [Batch, 3, 84, Time]
        math_vector: [Batch, 90]
        """
        # 1. Process visual features (Outputs [Batch, 512])
        visual_features = self.resnet(image_tensor)
        # Explicit flatten as a safety net ensuring it is 2D
        visual_features = visual_features.view(visual_features.size(0), -1)
        visual_features = self.visual_dropout(visual_features)
        
        # 2. Process mathematical features (Outputs [Batch, 128])
        math_features = self.math_mlp(math_vector)
        
        # 3. Apply independent L2 Normalization to prevent architectural bias
        visual_features = F.normalize(visual_features, p=2, dim=1)
        math_features = F.normalize(math_features, p=2, dim=1)
        
        # 4. Concatenate features along the feature dimension (Outputs [Batch, 640])
        fused = torch.cat((visual_features, math_features), dim=1)

        # 4. Process through final funnel (Outputs [Batch, 64])
        embedding = self.fusion_mlp(fused)
        
        # 5. Apply L2 Normalization so embeddings exist on a unit hypersphere
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
