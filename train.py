import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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

class LoopTripletDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.positives = self.df[self.df['label'] == 1].reset_index(drop=True)
        self.negatives = self.df[self.df['label'] == 0].reset_index(drop=True)
        
    def __len__(self):
        return len(self.positives)
        
    def _load_tensor_dict(self, filepath):
        tensor_dict = torch.load(filepath, weights_only=True)
        image = tensor_dict['image_tensor']
        target_length = 350
        current_length = image.shape[2]
        
        if current_length < target_length:
            import torch.nn.functional as F
            image = F.pad(image, (0, target_length - current_length), "constant", 0)
        elif current_length > target_length:
            image = image[:, :, :target_length]
        math_vector = tensor_dict['math_vector']
        
        image = torch.nan_to_num(image, nan=0.0)
        math_vector = torch.nan_to_num(math_vector, nan=0.0)
            
        return image, math_vector

    def __getitem__(self, idx):
        pos_row = self.positives.iloc[idx]
        image_anc, math_anc = self._load_tensor_dict(pos_row['file_A'])
        image_pos, math_pos = self._load_tensor_dict(pos_row['file_B'])
        
        neg_row = self.negatives.sample(1).iloc[0]
        image_neg, math_neg = self._load_tensor_dict(neg_row['file_B'])
        
        return image_anc, math_anc, image_pos, math_pos, image_neg, math_neg


def train():
    # 1. Setup Data
    print(">>> 1. Initializing Triplet Dataset and DataLoader...")
    dataset = LoopTripletDataset("train_pairs.csv")
    
    # We use num_workers=0 to safely handle the temporary file writing in the dataset
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"     -> Dataset loaded successfully with {len(dataset)} total pairs.")
    print(f"     -> Total Batches per Epoch: {len(dataloader)} (Batch Size: {batch_size})")
    
    print(">>> 1.5 Initializing Validation DataLoader...")
    val_dataset = LoopPairDataset("test_pairs.csv")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"     -> Validation Dataset loaded with {len(val_dataset)} pairs.")
    
    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f">>> 2. Neural Network dispatched to device: [{device.type.upper()}]")
    
    # 3. Initialize Model and Loss
    print(">>> 3. Initializing True Siamese Network and Triplet Margin Loss...")
    model = SiameseNetwork().to(device)
    criterion = torch.nn.TripletMarginLoss(margin=2.0, p=2).to(device)
    val_criterion = ContrastiveLoss(margin=2.0).to(device)
    
    # 4. Setup Optimizer
    # IMPORTANT: We filter out frozen parameters (layer1/2) so Adam doesn't waste compute tracking them
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # Lowered learning rate to 0.0001 and added weight decay to fight Overfitting
    optimizer = Adam(trainable_params, lr=0.0001, weight_decay=1e-4)
    
    # 5. Training Loop
    # Reduced to 15 epochs to prevent the model from memorizing the training data (Early Stopping)
    num_epochs = 15
    print("\n=======================================================")
    print(f"--- Starting Siamese Network Training ({num_epochs} Epochs) ---")
    print("=======================================================\n")
    
    total_start_time = time.time()
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        print(f"+++ EPOCH {epoch+1}/{num_epochs} STARTED +++")
        
        for batch_idx, (image_anc, math_anc, image_pos, math_pos, image_neg, math_neg) in enumerate(dataloader):
            # Send everything to the GPU/Device
            image_anc = image_anc.to(device)
            math_anc = math_anc.to(device)
            image_pos = image_pos.to(device)
            math_pos = math_pos.to(device)
            image_neg = image_neg.to(device)
            math_neg = math_neg.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward Pass: Push triplets through their independent towers
            embed_anc, embed_pos, embed_neg = model(image_anc, math_anc, image_pos, math_pos, image_neg, math_neg)
            
            # Calculate Triplet Loss
            loss = criterion(embed_anc, embed_pos, embed_neg)
            
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
        
        # --- Validation Pass ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image_A, math_A, image_B, math_B, label in val_dataloader:
                image_A, math_A = image_A.to(device), math_A.to(device)
                image_B, math_B = image_B.to(device), math_B.to(device)
                label = label.to(device)
                
                embed_A, embed_B = model(image_A, math_A, image_B, math_B)
                loss = val_criterion(embed_A, embed_B, label)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_dataloader)
        
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)
        
        epoch_duration = time.time() - epoch_start_time
        print(f"\n=== EPOCH {epoch+1} FINISHED ===")
        print(f"    -> Final Train Loss: {avg_loss:.4f} | Final Val Loss: {avg_val_loss:.4f}")
        print(f"    -> Epoch Duration: {epoch_duration:.1f} seconds\n")
        print("-" * 55 + "\n")
        
    # 6. Save Model State
    total_duration = (time.time() - total_start_time) / 60
    
    # Save to the models folder with a timestamp so we never overwrite
    os.makedirs("models", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = f"models/siamese_weights_{timestamp}.pth"
    
    torch.save(model.state_dict(), save_path)
    
    # 7. Generate Overfitting Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), history['train_loss'], label='Training Loss (Tracks 1-16)', color='blue', marker='o')
    plt.plot(range(1, num_epochs+1), history['val_loss'], label='Validation Loss (Tracks 17-20)', color='red', marker='x')
    plt.title('Training vs Validation Loss (The Concrete Ceiling)')
    plt.xlabel('Epochs')
    plt.ylabel('Contrastive Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = f"models/loss_curve_{timestamp}.png"
    plt.savefig(plot_path)
    
    print("=======================================================")
    print(f"TRAINING COMPLETE IN {total_duration:.2f} MINUTES!")
    print(f"Model weights successfully saved to: {save_path}")
    print(f"Loss curve successfully saved to: {plot_path}")
    print("=======================================================")

if __name__ == "__main__":
    train()
