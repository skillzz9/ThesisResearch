import os
# CRITICAL FIX: Prevent Numpy/Librosa from hijacking all 12 cores for a single track!
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob

# Import the core modules of our pipeline
from create_loops import process_track
from extract_features import extract_loop_features
from generate_pairs import generate_pair_dataset
from train import train

def run_master_pipeline():
    """
    The master orchestrator.
    Runs every phase of the pipeline from raw stems to a fully trained neural network.
    """
    root_dir = "/Users/hugoposthuma/Downloads/Thesis/babyslakh_16k"
    
    # 1. Discover all tracks (recursively, since they are inside train/, test/, validation/)
    tracks = sorted([d for d in glob.glob(os.path.join(root_dir, "**", "Track*"), recursive=True) if os.path.isdir(d)])
    
    if not tracks:
        print(f"CRITICAL ERROR: No track directories found in {root_dir}")
        return
        
    print("=================================================================")
    print(f" MASTER PIPELINE INITIATED: Found {len(tracks)} Tracks to Process")
    print("=================================================================\n")
    
    # 2. Iterate through each track to Slice and Extract
    print(f"\n{'='*50}")
    print(f" PHASE 1 & 2: Slicing & Feature Extraction (12-Core Parallel)")
    print(f"{'='*50}")
    
    import multiprocessing
    
    def process_single_track(track_dir):
        track_name = os.path.basename(track_dir)
        print(f"--> Processing [{track_name}]")
        process_track(track_dir)
        extract_loop_features(track_dir)
        return track_name

    print(f"🚀 RUNNING IN SEQUENTIAL VERBOSE MODE FOR LOCAL TESTING 🚀")
    
    for track_dir in tracks:
        print(f"\n==================================================")
        print(f" STARTING TRACK: {os.path.basename(track_dir)}")
        print(f"==================================================")
        import sys
        sys.stdout.flush() # Force MacOS to print immediately
        
        process_single_track(track_dir)
        
    # 3. Build the Metric Learning Dataset
    print("\n=================================================================")
    print(" PHASE 3: Generating Siamese Pair Dataset (dataset_pairs.csv)")
    print("=================================================================")
    generate_pair_dataset(root_dir, "dataset_pairs.csv")
    
    print("\n=================================================================")
    print(" PHASE 3.5: Splitting into 80/20 Train/Test Sets")
    print("=================================================================")
    from split_dataset import split_dataset
    split_dataset("dataset_pairs.csv")
    
    # 4. Train the Neural Network
    print("\n=================================================================")
    print(" PHASE 4: Booting PyTorch Training Loop")
    print("=================================================================")
    train()
    
    print("\n\n=================================================================")
    print(" FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=================================================================\n")

if __name__ == "__main__":
    run_master_pipeline()
