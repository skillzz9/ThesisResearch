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
    
    # MASSIVE SPEEDUP: Cap to a maximum of 40 loops per track to prevent overkill
    if len(wav_files) > 40:
        import random
        random.seed(42) # Deterministic shuffle
        wav_files = random.sample(wav_files, 40)
        
    print(f"Found {len(wav_files)} loops. Extracting features to .pt files...")
    
    for i, wav_path in enumerate(wav_files):
        # FAST RESUME: Check if all 7 shifts already exist to skip this loop instantly
        shifts = [-3, -2, -1, 0, 1, 2, 3]
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
