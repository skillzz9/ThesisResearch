import os
import glob
import torch
import librosa
import tempfile
import soundfile as sf
from left_eye.feature_extractor import VisualFeatureExtractor
from right_eye.pipeline import RightEyePipeline

def extract_loop_features(track_dir):
    """
    Reads every .wav loop, extracts the Left and Right Eye tensors,
    and saves them as a combined .pt file.
    """
    loops_dir = os.path.join(track_dir, "loops")
    vectors_dir = os.path.join(track_dir, "loop_vectors")
    
    os.makedirs(vectors_dir, exist_ok=True)
    
    # Initialize pipelines
    visual_extractor = VisualFeatureExtractor(sample_rate=22050)
    right_eye = RightEyePipeline()
    
    wav_files = sorted(glob.glob(os.path.join(loops_dir, "*.wav")))
    
    # --- THE GOLDEN MEASURE APPROACH ---
    # Group the wav files by their "place" (measure)
    from collections import defaultdict
    import random
    random.seed(42) # Deterministic
    
    places = defaultdict(list)
    for w in wav_files:
        # Expected format: track_00001_stem1_place01.wav
        basename = os.path.basename(w)
        parts = basename.replace(".wav", "").split("_")
        if len(parts) >= 4:
            place_id = parts[3] # "place01"
            places[place_id].append(w)
            
    # Find Golden Measures (measures where at least 3 instruments are playing)
    golden_measures = {p: files for p, files in places.items() if len(files) >= 3}
    
    # If no measure has 3+ instruments, fallback to measures with 2+ instruments
    if not golden_measures:
        golden_measures = {p: files for p, files in places.items() if len(files) >= 2}
        
    # Select just 2 specific Golden Measures to extract (Max Diversity, Extreme Speed)
    selected_measures = []
    if len(golden_measures) > 2:
        selected_keys = random.sample(list(golden_measures.keys()), 2)
        for k in selected_keys:
            selected_measures.extend(golden_measures[k])
    else:
        for files in golden_measures.values():
            selected_measures.extend(files)
            
    wav_files = selected_measures
    
    print(f"Found {len(wav_files)} loops in Golden Measures. Extracting features to .pt files...")
    
    for i, wav_path in enumerate(wav_files):
        # FAST RESUME: Limit shifts to just [-2, 0, 2] for 60% massive speedup
        shifts = [-2, 0, 2]
        all_exist = True
        for shift in shifts:
            basename = os.path.basename(wav_path).replace(".wav", f"_shift{shift}.pt")
            pt_path = os.path.join(vectors_dir, basename)
            if not os.path.exists(pt_path):
                all_exist = False
                break
                
        if all_exist:
            # Skip the heavy librosa/openSMILE CPU math!
            continue
            
        # Load the base audio file once
        y_base, sr = librosa.load(wav_path, sr=22050)
        
        for shift in shifts:
            basename = os.path.basename(wav_path).replace(".wav", f"_shift{shift}.pt")
            pt_path = os.path.join(vectors_dir, basename)
            
            # Skip this specific shift if it was partially completed before crashing
            if os.path.exists(pt_path):
                continue
                
            # 1. Pitch Shift the audio
            if shift != 0:
                y = librosa.effects.pitch_shift(y_base, sr=sr, n_steps=shift)
            else:
                y = y_base
                
            # 2. Extract Left Eye (Visual)
            stacked_features = visual_extractor.extract_and_stack(y)
            image_tensor = torch.tensor(stacked_features, dtype=torch.float32)
            
            # 3. Extract Right Eye (Mathematical)
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                sf.write(temp_path, y, sr)
                math_vector_np = right_eye.get_vector(temp_path)
                math_vector = torch.tensor(math_vector_np, dtype=torch.float32)
            finally:
                os.remove(temp_path)
            
            # 4. Save as a dictionary in a .pt file
            torch.save({
                'image_tensor': image_tensor,
                'math_vector': math_vector
            }, pt_path)
            
        if (i + 1) % 10 == 0:
            print(f"Extracted all shifts for {i + 1}/{len(wav_files)} loops...")

    print(f"Finished extracting! All .pt files saved to {vectors_dir}")

if __name__ == "__main__":
    extract_loop_features("/Users/hugoposthuma/Downloads/Thesis/babyslakh_16k/Track00001")
