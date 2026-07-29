import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import glob

# Import our architecture
from siamese import SiameseNetwork
from train import LoopPairDataset

def get_distances_and_labels(model, dataloader, device):
    all_distances = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (image_A, math_A, image_B, math_B, label) in enumerate(dataloader):
            image_A = image_A.to(device)
            math_A = math_A.to(device)
            image_B = image_B.to(device)
            math_B = math_B.to(device)
            label = label.to(device)
            
            # Forward pass
            output = model(image_A, math_A, image_B, math_B)
            
            # Handle model output structure
            if isinstance(output, tuple) and len(output) >= 2:
                embed_A, embed_B = output[0], output[1]
            else:
                embed_A, embed_B = output
                
            distances = F.pairwise_distance(embed_A, embed_B)
            
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(label.squeeze().cpu().numpy())
            
    return np.array(all_distances), np.array(all_labels)

def find_best_threshold(distances, labels):
    """
    Sweeps possible distance thresholds from 0.0 to 4.0 to find the one that 
    maximizes classification accuracy on the training set.
    """
    thresholds = np.linspace(0.0, 4.0, 401)
    best_acc = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        preds = (distances < thresh).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    return best_thresh, best_acc * 100

def print_distribution_stats(distances, labels, split_name):
    pos_dists = distances[labels == 1]
    neg_dists = distances[labels == 0]
    
    # Avoid errors if a batch had zero positives or negatives
    if len(pos_dists) > 0 and len(neg_dists) > 0:
        print(f"  [{split_name}] Positive Pairs Distance: mean={pos_dists.mean():.4f}, std={pos_dists.std():.4f}")
        print(f"  [{split_name}] Negative Pairs Distance: mean={neg_dists.mean():.4f}, std={neg_dists.std():.4f}")
    else:
        print(f"  [{split_name}] Missing positive or negative samples for distribution stats.")

def evaluate_honest():
    print("EVALUATING MODEL... PLEASE WAIT...")

    print(">>> 1. Initializing Siamese Network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    
    # Auto-load the newest weights from the models folder
    if os.path.exists("models/Attempt2.pth"):
        weights_path = "models/Attempt2.pth"
    elif os.path.exists("models/FirstAttempt.pth"):
        weights_path = "models/FirstAttempt.pth"
    else:
        weights_path = None
        
    if weights_path:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            print(f"     -> Successfully loaded newest weights from: {weights_path}")
        except Exception as e:
            print(f"WARNING: Failed to load {weights_path}. Error: {e}")
            return
    else:
        print("CRITICAL WARNING: No weights found in models/ directory. You must run train.py first!")
        return
    
    model.eval()
    
    print("\n>>> 2. Extracting Distances on Train Set (Practice Test)...")
    train_dataset = LoopPairDataset("train_pairs.csv")
    # Increased batch size slightly for faster evaluation speed
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    train_dists, train_labels = get_distances_and_labels(model, train_dataloader, device)
    
    # Fit threshold purely on training data to prevent Data Leakage
    best_thresh, train_model_acc = find_best_threshold(train_dists, train_labels)
    
    test_dataset = LoopPairDataset("test_pairs.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    test_dists, test_labels = get_distances_and_labels(model, test_dataloader, device)
    
    # -------------------------------------------------------------
    # THE HONEST TEST: Apply the untouched train threshold to test
    # -------------------------------------------------------------
    test_preds = (test_dists < best_thresh).astype(int)
    test_model_acc = accuracy_score(test_labels, test_preds) * 100
    
    # Per-class recall helps us verify the model isn't just blindly guessing 0 for everything
    pos_recall = recall_score(test_labels, test_preds, pos_label=1, zero_division=0) * 100
    neg_recall = recall_score(test_labels, test_preds, pos_label=0, zero_division=0) * 100
    
    # ROC-AUC (Threshold-Free metric for compatibility scoring)
    # Since low distance = high compatibility, we invert distances for ROC-AUC
    inverted_dists = -test_dists 
    roc_auc = roc_auc_score(test_labels, inverted_dists) * 100

    print("\n=======================================================")
    print("               EVALUATION RESULTS")
    print("=======================================================")
    print(f"Metric                          | Score")
    print(f"--------------------------------|-----------")
    print(f"Optimal Distance Threshold      | {best_thresh:.4f}")
    print(f"Seen Data Accuracy (Train)      | {train_model_acc:.2f}%")
    print(f"Unseen Data Accuracy (Test)     | {test_model_acc:.2f}%")
    print(f"Overfitting Gap                 | {train_model_acc - test_model_acc:.2f}%")
    print(f"Test Positive Recall (Matches)  | {pos_recall:.2f}%")
    print(f"Test Negative Recall (Clashes)  | {neg_recall:.2f}%")
    print(f"ROC-AUC Score                   | {roc_auc:.2f}%")
    print("=======================================================\n")

if __name__ == "__main__":
    evaluate_honest()
