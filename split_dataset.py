import pandas as pd

def split_dataset(input_csv="dataset_pairs.csv"):
    print(f"Loading master dataset: {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # We will identify the track number by looking at file_A
    # Example file_A: /path/to/Track00001/loop_vectors/track_00001_stem1_place01_shift0.pt
    # We can extract the track number by searching for "track_" and parsing the number
    
    def get_track_num(filepath):
        # find 'track_' in the filename
        basename = filepath.split('/')[-1]
        # basename is e.g. track_00001_stem1_place01_shift0.pt
        parts = basename.split('_')
        for i, part in enumerate(parts):
            if part == "track":
                return int(parts[i+1])
        return 0

    print("Splitting dataset into 80% Train and 20% Test dynamically...")
    
    # Apply the function to create a temporary track column
    df['track_num'] = df['file_A'].apply(get_track_num)
    
    # Calculate dynamic cutoff (80% of max track num)
    max_track = df['track_num'].max()
    cutoff = int(max_track * 0.8)
    
    # Split the data based on the dynamic track number
    train_df = df[df['track_num'] <= cutoff].drop(columns=['track_num'])
    test_df = df[df['track_num'] > cutoff].drop(columns=['track_num'])
    
    train_csv = "train_pairs.csv"
    test_csv = "test_pairs.csv"
    
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print("\n--- Split Complete ---")
    print(f"Training Pairs (Tracks <= {cutoff}): {len(train_df)}")
    print(f"Testing Pairs (Tracks > {cutoff}): {len(test_df)}")
    print(f"Saved to {train_csv} and {test_csv}")

if __name__ == "__main__":
    split_dataset()
