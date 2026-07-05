import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Import our custom architecture components
from towers import MusicTower
from siamese import SiameseNetwork
from loss import ContrastiveLoss

# Import the base feature extractors from our modular pipelines
from left_eye.feature_extractor import VisualFeatureExtractor
from right_eye.pipeline import RightEyePipeline


class LoopPairDataset(Dataset):
    def __init__(self, csv_file):
        """
        Reads the dataset map which points to the pre-extracted .pt files.
        """
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.df)
        
    def _load_tensor_dict(self, filepath):
        """
        Loads the .pt file containing the image_tensor and math_vector directly.
        Pads the time dimension so that all batches are perfectly uniform.
        """
        # Load the dictionary from disk
        tensor_dict = torch.load(filepath, weights_only=True)
        
        # CQT / Chromagrams have different time lengths based on the track's BPM.
        # We pad the time dimension (dim 2) to a fixed width (e.g., 350 frames) 
        # so PyTorch can stack them into a batch. 350 frames covers loops up to ~8 seconds.
        image = tensor_dict['image_tensor']
        target_length = 350
        current_length = image.shape[2]
        
        if current_length < target_length:
            # Pad the right side of the time dimension with zeros
            import torch.nn.functional as F
            image = F.pad(image, (0, target_length - current_length), "constant", 0)
        elif current_length > target_length:
            # Crop it if it happens to be unusually long
            image = image[:, :, :target_length]
        math_vector = tensor_dict['math_vector']
        
        # Sanitize data: If openSMILE hit a silent patch and returned NaN, convert it to 0.0
        image = torch.nan_to_num(image, nan=0.0)
        math_vector = torch.nan_to_num(math_vector, nan=0.0)
            
        return image, math_vector

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load Loop A (Pitch shifting is currently bypassed because we pre-extracted)
        image_A, math_A = self._load_tensor_dict(row['file_A'])
        
        # Load Loop B
        image_B, math_B = self._load_tensor_dict(row['file_B'])
        
        # Grab Label
        label = torch.tensor([row['label']], dtype=torch.float32)
        
        return image_A, math_A, image_B, math_B, label


def train():
    # 1. Setup Data
    print(">>> 1. Initializing Dataset and DataLoader...")
    dataset = LoopPairDataset("dataset_pairs.csv")
    
    # We use num_workers=0 to safely handle the temporary file writing in the dataset
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"     -> Dataset loaded successfully with {len(dataset)} total pairs.")
    print(f"     -> Total Batches per Epoch: {len(dataloader)} (Batch Size: {batch_size})")
    
    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f">>> 2. Neural Network dispatched to device: [{device.type.upper()}]")
    
    # 3. Initialize Model and Loss
    print(">>> 3. Initializing True Siamese Network and Contrastive Loss...")
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=2.0).to(device)
    
    # 4. Setup Optimizer
    # IMPORTANT: We filter out frozen parameters (layer1/2) so Adam doesn't waste compute tracking them
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # Lowered learning rate to 0.0001 to prevent Exploding Gradients
    optimizer = Adam(trainable_params, lr=0.0001)
    
    # 5. Training Loop
    # We increased this to 25 epochs so the network has enough time to fully map out 
    # the complex rules of music theory (especially the Hard Negatives).
    num_epochs = 25
    print("\n=======================================================")
    print(f"--- Starting Siamese Network Training ({num_epochs} Epochs) ---")
    print("=======================================================\n")
    
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        print(f"+++ EPOCH {epoch+1}/{num_epochs} STARTED +++")
        
        for batch_idx, (image_A, math_A, image_B, math_B, label) in enumerate(dataloader):
            # Send everything to the GPU/Device
            image_A = image_A.to(device)
            math_A = math_A.to(device)
            image_B = image_B.to(device)
            math_B = math_B.to(device)
            label = label.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward Pass: Push both loops through their independent towers
            embed_A, embed_B = model(image_A, math_A, image_B, math_B)
            
            # Calculate Metric Distance and Loss
            loss = criterion(embed_A, embed_B, label)
            
            # Backward Pass
            loss.backward()
            
            # Gradient Clipping: Prevents the math from exploding into NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            # Optimize
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print live progress every 5 batches to monitor health closely
            if batch_idx % 5 == 0 or batch_idx == len(dataloader) - 1:
                running_avg_loss = epoch_loss / (batch_idx + 1)
                elapsed_batch_time = time.time() - epoch_start_time
                print(f"    [Batch {batch_idx:4d}/{len(dataloader):4d}] | "
                      f"Step Loss: {loss.item():.4f} | "
                      f"Running Epoch Avg: {running_avg_loss:.4f} | "
                      f"Time Elapsed: {elapsed_batch_time:.1f}s")
                
        # End of Epoch logging
        avg_loss = epoch_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start_time
        print(f"\n=== EPOCH {epoch+1} FINISHED ===")
        print(f"    -> Final Average Loss: {avg_loss:.4f}")
        print(f"    -> Epoch Duration: {epoch_duration:.1f} seconds\n")
        print("-" * 55 + "\n")
        
    # 6. Save Model State
    total_duration = (time.time() - total_start_time) / 60
    save_path = "siamese_weights.pth"
    torch.save(model.state_dict(), save_path)
    
    print("=======================================================")
    print(f"TRAINING COMPLETE IN {total_duration:.2f} MINUTES!")
    print(f"Model weights successfully saved to: {save_path}")
    print("=======================================================")

if __name__ == "__main__":
    train()
