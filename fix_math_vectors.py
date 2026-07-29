import os
import glob
import torch
import librosa
import tempfile
import soundfile as sf
import sys
from right_eye.pipeline import RightEyePipeline

def fix_math_vectors():
    print("Starting Math Vector Fixer...")
    root_dir = "/Users/hugoposthuma/Downloads/Thesis/babyslakh_16k"
    tracks = sorted([d for d in glob.glob(os.path.join(root_dir, "Track*")) if os.path.isdir(d)])
    
    right_eye = RightEyePipeline()
    shifts = [-3, -2, -1, 0, 1, 2, 3]
    
    total_files = 0
    fixed_files = 0
    
    for track_dir in tracks:
        track_name = os.path.basename(track_dir)
        loops_dir = os.path.join(track_dir, "loops")
        vectors_dir = os.path.join(track_dir, "loop_vectors")
        
        wav_files = sorted(glob.glob(os.path.join(loops_dir, "*.wav")))
        if not wav_files:
            continue
            
        print(f"Fixing {track_name} ({len(wav_files)} original loops)...")
        
        for wav_path in wav_files:
            # Load original wav once
            y_base, sr = librosa.load(wav_path, sr=22050)
            
            for shift in shifts:
                basename = os.path.basename(wav_path).replace(".wav", f"_shift{shift}.pt")
                pt_path = os.path.join(vectors_dir, basename)
                
                if not os.path.exists(pt_path):
                    continue
                    
                total_files += 1
                
                # Pitch shift the audio
                if shift != 0:
                    y = librosa.effects.pitch_shift(y_base, sr=sr, n_steps=shift)
                else:
                    y = y_base
                    
                # Extract clean math vector
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    sf.write(temp_path, y, sr)
                    math_vector_np = right_eye.get_vector(temp_path)
                    math_vector = torch.tensor(math_vector_np, dtype=torch.float32)
                finally:
                    os.remove(temp_path)
                    
                # Load existing PT file and ONLY overwrite the math_vector
                data = torch.load(pt_path, weights_only=True)
                data['math_vector'] = math_vector
                torch.save(data, pt_path)
                
                fixed_files += 1
                
    print(f"Successfully fixed {fixed_files}/{total_files} .pt files.")

if __name__ == "__main__":
    fix_math_vectors()
