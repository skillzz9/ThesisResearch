import os
import glob
import mido
import scipy.io.wavfile as wavfile
import numpy as np

def extract_bpm_from_midi(midi_path):
    """Extracts the exact BPM from a MIDI file's tempo map."""
    mid = mido.MidiFile(midi_path)
    tempos = [msg.tempo for track in mid.tracks for msg in track if msg.type == 'set_tempo']
    if not tempos:
        raise ValueError(f"No tempo found in {midi_path}")
    return mido.tempo2bpm(tempos[0])

def process_track(track_dir):
    """Processes a track directory by slicing its stems into 8-beat loops."""
    stems_dir = os.path.join(track_dir, "stems")
    midi_path = os.path.join(track_dir, "all_src.mid")
    loops_dir = os.path.join(track_dir, "loops")
    
    if not os.path.exists(loops_dir):
        os.makedirs(loops_dir)
        
    print(f"--- Processing Track: {track_dir} ---")
    
    # 1. Extract BPM & calculate slice duration
    bpm = extract_bpm_from_midi(midi_path)
    print(f"Detected BPM: {bpm}")
    
    seconds_per_beat = 60.0 / bpm
    loop_duration_sec = seconds_per_beat * 8.0  # 8 beats = 2 bars
    print(f"Calculated 8-beat loop duration: {loop_duration_sec:.3f} seconds\n")
    
    # 2. Get the track name for formatting (e.g. Track00001 -> track_00001)
    track_folder_name = os.path.basename(track_dir.strip("/"))
    track_id_str = track_folder_name.lower().replace("track", "track_")
    
    # 3. Find and slice stems
    wav_files = sorted(glob.glob(os.path.join(stems_dir, "*.wav")))
    if not wav_files:
        print(f"No .wav files found in {stems_dir}")
        return
        
    for wav_file in wav_files:
        stem_filename = os.path.basename(wav_file)
        stem_name = stem_filename.replace(".wav", "")
        
        # Convert "S01" -> "stem1", "S10" -> "stem10"
        if stem_name.startswith("S"):
            try:
                stem_num = int(stem_name[1:])
                stem_id_str = f"stem{stem_num}"
            except ValueError:
                stem_id_str = stem_name
        else:
            stem_id_str = stem_name
            
        print(f"Slicing {stem_filename} -> {stem_id_str}...")
        
        # Load audio (assuming it's a standard uncompressed WAV)
        sample_rate, audio_data = wavfile.read(wav_file)
        
        # Calculate exactly how many samples represent our loop duration
        samples_per_loop = int(round(loop_duration_sec * sample_rate))
        total_samples = len(audio_data)
        
        # Determine how many full loops fit into the audio
        num_loops = total_samples // samples_per_loop
        
        for i in range(num_loops):
            start_idx = i * samples_per_loop
            end_idx = start_idx + samples_per_loop
            
            # Slice exactly 96000 samples (if 16kHz at 80BPM)
            chunk = audio_data[start_idx:end_idx]
            
            # Check for pure silence. If the chunk is completely silent, skip saving it.
            # We cast to float to safely use abs() avoiding int16 overflow issues.
            if np.max(np.abs(chunk.astype(float))) < 1e-4:
                continue
            
            # Place index (1-based, zero-padded to 2 digits, e.g., 01, 02, 10)
            # This represents the absolute grid location, preserving compatibility even if earlier loops were silent.
            place = i + 1
            loop_filename = f"{track_id_str}_{stem_id_str}_place{place:02d}.wav"
            loop_path = os.path.join(loops_dir, loop_filename)
            
            # Save the new chunk
            wavfile.write(loop_path, sample_rate, chunk)
            
    print(f"\nFinished! All loops successfully saved to:\n{loops_dir}")

if __name__ == "__main__":
    # Test path for Track00001
    track_path = "/Users/hugoposthuma/Downloads/Thesis/babyslakh_16k/Track00001"
    process_track(track_path)
