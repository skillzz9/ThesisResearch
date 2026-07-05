import os
import glob
import torch
from main import process_audio

stems_dir = "/Users/hugoposthuma/Downloads/Thesis/babyslakh_16k/Track00001/stems"
vectors_dir = "/Users/hugoposthuma/Downloads/Thesis/babyslakh_16k/Track00001/vectors"

# Create vectors directory if it doesn't exist
os.makedirs(vectors_dir, exist_ok=True)

# Find all wav files in the stems directory
wav_files = glob.glob(os.path.join(stems_dir, "*.wav"))

if not wav_files:
    print("No .wav files found in", stems_dir)

for wav_file in sorted(wav_files):
    filename = os.path.basename(wav_file)
    vector_path = os.path.join(vectors_dir, filename.replace(".wav", ".pt"))
    
    print(f"\\n{'='*40}\\nProcessing {filename}...")
    try:
        final_vector = process_audio(wav_file)
        torch.save(final_vector, vector_path)
        print(f"Successfully saved vector to {vector_path}")
    except Exception as e:
        print(f"Failed to process {filename}: {e}")

print(f"\\n{'='*40}\\nFinished processing all stems in Track00001.")
