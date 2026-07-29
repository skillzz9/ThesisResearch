import os
import random
import glob
import yaml
import pandas as pd
from itertools import combinations

def parse_filename(filepath, base_dir):
    """
    Parses a filename like 'track_00001_stem1_place01_shift0.pt'
    Returns a dictionary with the track, stem, place, and its mapped instrument category.
    """
    basename = os.path.basename(filepath).replace("_shift0.pt", "")
    parts = basename.split("_")
    if len(parts) >= 4:
        # Construct exact track folder name, e.g. "Track00001"
        track_folder_name = f"Track{parts[1]}"
        track_folder = os.path.join(base_dir, track_folder_name)
        
        stem_raw = parts[2] # e.g. 'stem1'
        
        # Convert 'stem1' to 'S01' for YAML lookup
        stem_num = stem_raw.replace("stem", "")
        stem_key = f"S{int(stem_num):02d}"
        
        category = "Unknown"
        yaml_path = os.path.join(track_folder, "metadata.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                try:
                    inst_class = data['stems'][stem_key]['inst_class']
                    if inst_class == "Drums":
                        category = "Drums"
                    elif inst_class == "Bass":
                        category = "Bass"
                    else:
                        # Pianos, Organs, Guitars, Strings all become "Melody/Synth"
                        category = "Melody" 
                except:
                    pass
        
        return {
            "filepath": filepath,
            "track": track_folder_name,
            "stem": stem_raw,
            "place": parts[3],
            "category": category
        }
    return None

def is_useful_combination(cat_a, cat_b):
    """
    Only allow the specific pairings you requested.
    """
    pair = set([cat_a, cat_b])
    
    allowed = [
        set(["Drums", "Bass"]),      # Drums with Bass
        set(["Melody", "Bass"]),     # Synth with Bass
        set(["Melody", "Melody"]),   # Synth with Vocals/Other Melodies
        set(["Melody", "Drums"])     # Bonus: Melody with Drums
    ]
    return pair in allowed

def generate_pair_dataset(vectors_dir, output_csv="dataset_pairs.csv"):
    all_files = glob.glob(os.path.join(vectors_dir, "**/*_shift0.pt"), recursive=True)
    
    parsed_data = [parse_filename(f, vectors_dir) for f in all_files]
    parsed_data = [x for x in parsed_data if x is not None]
    
    df = pd.DataFrame(parsed_data)
    
    if df.empty:
        print("No valid loop files found!")
        return

    positives = []
    negatives = []
    
    print(f"Found {len(df)} loop files. Generating targeted, useful pairs...")

    # Group by track and place
    grouped = df.groupby(["track", "place"])
    for (track, place), group in grouped:
        # Get all valid stems in this exact measure
        stems_info = group.to_dict('records')
        
        if len(stems_info) >= 2:
            all_combos = list(combinations(stems_info, 2))
            
            valid_combos = []
            for a, b in all_combos:
                if is_useful_combination(a["category"], b["category"]):
                    valid_combos.append((a["filepath"], b["filepath"]))
            
            # Max 5 random combinations per song-block to maximize diversity but increase volume
            if len(valid_combos) > 5:
                valid_combos = random.sample(valid_combos, 5)
                
            for file_a, file_b in valid_combos:
                # 1. Standard Positive
                positives.append({"file_A": file_a, "shift_A": 0, "file_B": file_b, "shift_B": 0, "label": 1})
                
                # 2. Augmented Positive
                sync_shift = random.choice([-3, -2, -1, 1, 2, 3])
                positives.append({"file_A": file_a, "shift_A": sync_shift, "file_B": file_b, "shift_B": sync_shift, "label": 1})
                
                # 3. Hard Negative: Pitch-Clash
                clash_shift = random.choice([-2, -1, 1, 2])
                negatives.append({"file_A": file_a, "shift_A": 0, "file_B": file_b, "shift_B": clash_shift, "label": 0})

    # 4. Hard Negative: Structure-Clash (Same Track, Different Place)
    track_grouped = df.groupby("track")
    for track, group in track_grouped:
        places = group["place"].unique()
        if len(places) >= 2:
            for _ in range(len(group) * 2): 
                p1, p2 = random.sample(list(places), 2)
                row_a = group[group["place"] == p1].sample(1).iloc[0]
                row_b = group[group["place"] == p2].sample(1).iloc[0]
                
                # Ensure useful combination
                if is_useful_combination(row_a["category"], row_b["category"]):
                    negatives.append({"file_A": row_a["filepath"], "shift_A": 0, "file_B": row_b["filepath"], "shift_B": 0, "label": 0})


    # Cross-Track Easy Negatives (Targeted)
    tracks = df["track"].unique()
    if len(tracks) >= 2:
        for _ in range(len(positives) * 2): 
            t1, t2 = random.sample(list(tracks), 2)
            
            # Grab random stems from each track
            row_a = df[df["track"] == t1].sample(1).iloc[0]
            row_b = df[df["track"] == t2].sample(1).iloc[0]
            
            # Only append if they are a useful combination
            if is_useful_combination(row_a["category"], row_b["category"]):
                negatives.append({
                    "file_A": row_a["filepath"], "shift_A": 0,
                    "file_B": row_b["filepath"], "shift_B": 0,
                    "label": 0
                })

    # Balance 50/50
    random.shuffle(positives)
    random.shuffle(negatives)
    
    min_count = min(len(positives), len(negatives))
    final_dataset = positives[:min_count] + negatives[:min_count]
    random.shuffle(final_dataset)
    
    # Map shifts to filenames
    for entry in final_dataset:
        entry["file_A"] = entry["file_A"].replace("_shift0.pt", f"_shift{entry['shift_A']}.pt")
        entry["file_B"] = entry["file_B"].replace("_shift0.pt", f"_shift{entry['shift_B']}.pt")
        
    pd.DataFrame(final_dataset).to_csv(output_csv, index=False)
    
    print(f"\n--- Targeted Dataset Generation Complete ---")
    print(f"Total Pairs: {len(final_dataset)}")
    print(f"Positives: {min_count} | Negatives: {min_count}")
    print(f"Saved to: {output_csv}")

if __name__ == "__main__":
    target_directory = "/Users/hugoposthuma/Downloads/Thesis/babyslakh_16k"
    generate_pair_dataset(target_directory, "dataset_pairs.csv")
