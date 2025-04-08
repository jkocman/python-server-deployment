import librosa
import numpy as np
import tempfile
import os

def music_analysis(file_like):
    # Uložení do dočasného souboru
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
        tmp.write(file_like.read())
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path)
        print(f"✅ Loaded audio with {len(y)} samples, sample rate: {sr}")
    except Exception as e:
        print(f"❌ ERROR: Nepodařilo se načíst audio: {e}")
        os.remove(tmp_path)
        raise e  # necháme výjimku pokračovat

    # Odstraníme dočasný soubor
    os.remove(tmp_path)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_value = float(tempo)
    length = librosa.get_duration(y=y, sr=sr)
    if length >= 90:
        length = 90

    rms = librosa.feature.rms(y=y)[0]
    window_size = 50
    rms_smooth = np.convolve(rms, np.ones(window_size) / window_size, mode='same')

    median_rms = np.median(rms_smooth)
    low_threshold = median_rms * 0.8
    medium_threshold = median_rms * 1.0
    high_threshold = median_rms * 1.2

    segment_duration = 10
    total_duration = len(y) / sr
    num_segments = int(np.ceil(total_duration / segment_duration))

    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)

        start_frame = librosa.time_to_frames(start_time, sr=sr)
        end_frame = librosa.time_to_frames(end_time, sr=sr)
        segment_rms = rms_smooth[start_frame:end_frame]

        avg_rms = np.mean(segment_rms)

        intensity = "low"
        if avg_rms > high_threshold:
            intensity = "high"
        elif avg_rms > medium_threshold:
            intensity = "medium"

        segments.append({
            "start": start_time,
            "end": end_time,
            "intensity": intensity,
            "rms": avg_rms
        })

    rms_tresholds = [
        f"high_threshold {high_threshold}",
        f"medium_threshold {medium_threshold}",
        f"low_threshold {low_threshold}"
    ]

    return tempo_value, length, segments, rms_tresholds