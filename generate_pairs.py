import os
import random
import glob
import pandas as pd
from itertools import combinations

def parse_filename(filepath):
    """
    Parses a filename like 'track_00001_stem1_place01_shift0.pt'
    Returns a dictionary with the track, stem, and place.
    """
    basename = os.path.basename(filepath).replace("_shift0.pt", "")
    parts = basename.split("_")
    if len(parts) >= 4:
        return {
            "filepath": filepath,
            "filename": basename,
            "track": f"{parts[0]}_{parts[1]}",  # e.g., 'track_00001'
            "stem": parts[2],                   # e.g., 'stem1'
            "place": parts[3]                   # e.g., 'place01'
        }
    return None

def generate_pair_dataset(vectors_dir, output_csv="dataset_pairs.csv"):
    # Find all base (_shift0) .pt files in vectors_dir
    all_files = glob.glob(os.path.join(vectors_dir, "**/*_shift0.pt"), recursive=True)
    
    parsed_data = [parse_filename(f) for f in all_files if parse_filename(f) is not None]
    df = pd.DataFrame(parsed_data)
    
    if df.empty:
        print(f"No valid loop files found in {loops_dir}!")
        return

    positives = []
    negatives = []
    
    print(f"Found {len(df)} loop files. Generating pairs...")

    # Group by track and place for Positives & Pitch-Clashes
    grouped = df.groupby(["track", "place"])
    for (track, place), group in grouped:
        stems = group["filepath"].tolist()
        if len(stems) >= 2:
            # Get every combination of different stems playing at the exact same time
            all_combos = list(combinations(stems, 2))
            
            # Hard Capping: Max 5 random combinations per song-block to prevent O(N^2) explosion
            if len(all_combos) > 5:
                all_combos = random.sample(all_combos, 5)
                
            for file_a, file_b in all_combos:
                
                # 1. Standard Positive (Label 1, No Pitch Shift)
                positives.append({
                    "file_A": file_a, "shift_A": 0,
                    "file_B": file_b, "shift_B": 0,
                    "label": 1
                })
                
                # 2. Augmented Positive (Label 1, Synchronized Pitch Shift)
                # Shift both equally so they still harmonically match
                sync_shift = random.choice([-3, -2, -1, 1, 2, 3])
                positives.append({
                    "file_A": file_a, "shift_A": sync_shift,
                    "file_B": file_b, "shift_B": sync_shift,
                    "label": 1
                })
                
                # 3. Hard Negative: Pitch-Clash (Label 0, Asymmetrical Pitch Shift)
                # Shift only one stem to create severe dissonance while tempo/groove match perfectly
                clash_shift = random.choice([-2, -1, 1, 2])
                negatives.append({
                    "file_A": file_a, "shift_A": 0,
                    "file_B": file_b, "shift_B": clash_shift,
                    "label": 0
                })

    # 4. Hard Negative: Structure-Clash (Label 0, Same Track, Different Place)
    track_grouped = df.groupby("track")
    for track, group in track_grouped:
        places = group["place"].unique()
        if len(places) >= 2:
            # Generate a solid amount of structural clashes
            for _ in range(len(group) * 2): 
                p1, p2 = random.sample(list(places), 2)
                file_a = random.choice(group[group["place"] == p1]["filepath"].tolist())
                file_b = random.choice(group[group["place"] == p2]["filepath"].tolist())
                
                # Avoid accidentally pairing the exact same stem from different places if you only want 
                # cross-stem structural clashes, but even the same stem from different places is a clash!
                negatives.append({
                    "file_A": file_a, "shift_A": 0,
                    "file_B": file_b, "shift_B": 0,
                    "label": 0
                })

    # 5. Easy Negatives (Label 0, Random Tracks)
    tracks = df["track"].unique()
    if len(tracks) >= 2:
        # Generate enough to bulk up the negatives
        for _ in range(len(positives)): 
            t1, t2 = random.sample(list(tracks), 2)
            file_a = random.choice(df[df["track"] == t1]["filepath"].tolist())
            file_b = random.choice(df[df["track"] == t2]["filepath"].tolist())
            negatives.append({
                "file_A": file_a, "shift_A": 0,
                "file_B": file_b, "shift_B": 0,
                "label": 0
            })
    else:
        print("Notice: Only 1 track found. Skipping 'Easy Negatives' (cross-track pairs).")

    # --- Balance the Dataset (Strict 50/50 Split) ---
    random.shuffle(positives)
    random.shuffle(negatives)
    
    # Cut down the larger list to match the smaller list exactly
    min_count = min(len(positives), len(negatives))
    
    balanced_positives = positives[:min_count]
    balanced_negatives = negatives[:min_count]
    
    final_dataset = balanced_positives + balanced_negatives
    
    # Final shuffle so 1s and 0s are completely mixed
    random.shuffle(final_dataset)
    
    # We need to map the selected shift back into the filepath since we are using pre-extracted shifted .pt files
    for entry in final_dataset:
        entry["file_A"] = entry["file_A"].replace("_shift0.pt", f"_shift{entry['shift_A']}.pt")
        entry["file_B"] = entry["file_B"].replace("_shift0.pt", f"_shift{entry['shift_B']}.pt")
        
    # Export
    df_final = pd.DataFrame(final_dataset)
    df_final.to_csv(output_csv, index=False)
    
    print(f"\n--- Dataset Generation Complete ---")
    print(f"Total Pairs: {len(df_final)}")
    print(f"Positives (Label 1): {min_count}")
    print(f"Negatives (Label 0): {min_count}")
    print(f"Saved directly to: {output_csv}")

if __name__ == "__main__":
    # Pointing it to the root babyslakh_16k folder will let it recursively find all track loop_vectors
    target_directory = "/Users/hugoposthuma/Downloads/Thesis/babyslakh_16k"
    generate_pair_dataset(target_directory, "dataset_pairs.csv")
