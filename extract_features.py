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
    print(f"Found {len(wav_files)} loops. Extracting features to .pt files...")
    
    for i, wav_path in enumerate(wav_files):
        # Load the base audio file once
        y_base, sr = librosa.load(wav_path, sr=22050)
        
        # We will generate features for standard pitch (0) and augmented pitches
        shifts = [-3, -2, -1, 0, 1, 2, 3]
        
        for shift in shifts:
            # 1. Pitch Shift the audio
            if shift != 0:
                y = librosa.effects.pitch_shift(y_base, sr=sr, n_steps=shift)
            else:
                y = y_base
                
            # 2. Extract Left Eye (Visual)
            stacked_features = visual_extractor.extract_and_stack(y)
            image_tensor = torch.tensor(stacked_features, dtype=torch.float32)
            
            # 3. Extract Right Eye (Mathematical)
            # Write shifted audio to temp file so Right Eye can read it
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                sf.write(temp_path, y, sr)
                math_vector_np = right_eye.get_vector(temp_path)
                math_vector = torch.tensor(math_vector_np, dtype=torch.float32)
                
                # FIX: Removed the bad cross-feature z-score normalization. 
                # Raw features will now be saved to disk, and column-wise scaling will happen during training.
            finally:
                os.remove(temp_path)
            
            # 4. Save as a dictionary in a .pt file
            # Format: track_00001_stem1_place01_shift3.pt
            basename = os.path.basename(wav_path).replace(".wav", f"_shift{shift}.pt")
            pt_path = os.path.join(vectors_dir, basename)
            
            torch.save({
                'image_tensor': image_tensor,
                'math_vector': math_vector
            }, pt_path)
            
        if (i + 1) % 10 == 0:
            print(f"Extracted all shifts for {i + 1}/{len(wav_files)} loops...")

    print(f"Finished extracting! All .pt files saved to {vectors_dir}")

if __name__ == "__main__":
    extract_loop_features("/Users/hugoposthuma/Downloads/Thesis/babyslakh_16k/Track00001")
