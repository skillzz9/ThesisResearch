import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Import our architecture
from siamese import SiameseNetwork
from train import LoopPairDataset

def evaluate_and_find_best_threshold():
    """
    Evaluates the trained Siamese Network and mathematically sweeps
    all possible thresholds to find the absolute maximum accuracy.
    """
    print(">>> 1. Initializing Siamese Network for Advanced Evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = SiameseNetwork().to(device)
    
    weights_path = "siamese_weights.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"     -> Successfully loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: {weights_path} not found.")
        return

    model.eval()

    print("\n>>> 2. Loading Dataset...")
    dataset = LoopPairDataset("dataset_pairs.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_distances = []
    all_labels = []
    
    print("\n=======================================================")
    print("--- Sweeping Dataset (This will take a minute) ---")
    print("=======================================================\n")
    
    with torch.no_grad():
        for batch_idx, (image_A, math_A, image_B, math_B, label) in enumerate(dataloader):
            image_A = image_A.to(device)
            math_A = math_A.to(device)
            image_B = image_B.to(device)
            math_B = math_B.to(device)
            
            # Get Embeddings and Distance
            embed_A, embed_B = model(image_A, math_A, image_B, math_B)
            distances = F.pairwise_distance(embed_A, embed_B)
            
            # Store them in CPU memory to analyze later
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(label.squeeze().numpy())
            
            if batch_idx % 100 == 0:
                print(f"    [Batch {batch_idx:4d}/{len(dataloader):4d}] Processed...")

    # Convert to numpy arrays for fast math
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)
    
    print("\n>>> 3. Calculating Optimal Threshold...")
    
    best_accuracy = 0.0
    best_threshold = 0.0
    
    # We will test every decimal threshold from 0.01 to 1.50
    test_thresholds = np.arange(0.01, 1.50, 0.01)
    
    for t in test_thresholds:
        # If distance < t, predict 1, else 0
        predictions = (all_distances < t).astype(float)
        
        # Calculate accuracy for this specific threshold
        correct = (predictions == all_labels).sum()
        accuracy = (correct / len(all_labels)) * 100
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t

    print("\n=======================================================")
    print(f" FINAL OPTIMIZED ACCURACY: {best_accuracy:.2f}%")
    print(f" PERFECT THRESHOLD CUTOFF: {best_threshold:.2f}")
    print("=======================================================")
    print("\nWhat this means: The network naturally decided that any two loops")
    print(f"with a distance less than {best_threshold:.2f} are compatible, and anything")
    print(f"greater than {best_threshold:.2f} is a clash!")

if __name__ == "__main__":
    evaluate_and_find_best_threshold()
