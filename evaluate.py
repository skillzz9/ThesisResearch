import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Import our architecture
from siamese import SiameseNetwork
from train import LoopPairDataset

def evaluate_model():
    print(">>> 1. Initializing Siamese Network for Evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = SiameseNetwork().to(device)
    
    # Load the trained weights
    weights_path = "siamese_weights.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"     -> Successfully loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: {weights_path} not found. You must train the model first!")
        return

    model.eval() # Set model to evaluation mode (turns off dropout/batchnorm updates)

    print("\n>>> 2. Loading Dataset (Testing on Training Data)...")
    dataset = LoopPairDataset("dataset_pairs.csv")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # We will track our accuracy
    correct_predictions = 0
    total_predictions = 0
    
    # In contrastive loss with margin=1.0, distance < 0.5 usually means they are similar (Label 1)
    # distance >= 0.5 means they are dissimilar (Label 0)
    threshold = 0.5 
    
    print("\n=======================================================")
    print("--- Starting Evaluation ---")
    print("=======================================================\n")
    
    with torch.no_grad(): # Turn off gradients to save massive amounts of RAM and time
        for batch_idx, (image_A, math_A, image_B, math_B, label) in enumerate(dataloader):
            image_A = image_A.to(device)
            math_A = math_A.to(device)
            image_B = image_B.to(device)
            math_B = math_B.to(device)
            label = label.to(device)
            
            # Get Embeddings
            embed_A, embed_B = model(image_A, math_A, image_B, math_B)
            
            # Calculate Euclidean Distance between the embeddings
            distances = F.pairwise_distance(embed_A, embed_B)
            
            # Predict: If distance < threshold, we predict 1. Otherwise 0.
            predictions = (distances < threshold).float()
            
            # Compare predictions to actual labels
            # label shape is [16, 1], predictions shape is [16]
            label_flat = label.squeeze() 
            correct_predictions += (predictions == label_flat).sum().item()
            total_predictions += len(label_flat)
            
            if batch_idx % 100 == 0:
                current_acc = (correct_predictions / total_predictions) * 100
                print(f"    [Batch {batch_idx:4d}/{len(dataloader):4d}] | Current Accuracy: {current_acc:.2f}%")
                
            # Optional: Stop early after 500 batches so we don't wait forever just to get a metric
            if batch_idx == 500:
                print("    -> Stopping early after 500 batches for a quick score.")
                break

    final_accuracy = (correct_predictions / total_predictions) * 100
    print("\n=======================================================")
    print(f" FINAL EVALUATION ACCURACY: {final_accuracy:.2f}%")
    print("=======================================================")

if __name__ == "__main__":
    evaluate_model()
