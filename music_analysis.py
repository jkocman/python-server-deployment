import librosa
import numpy as np

def music_analysis(file_like):
    try:
        y, sr = librosa.load(file_like, sr=22050, mono=True, duration=30)
        print(f"✅ Loaded audio with {len(y)} samples, sample rate: {sr}")
    except Exception as e:
        print(f"❌ ERROR: Nepodařilo se načíst audio: {e}")
        return "Error loading audio", 500

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except Exception as e:
        print(f"❌ ERROR tempo: {e}")
        tempo = 0.0

    tempo_value = float(tempo)
    length = min(librosa.get_duration(y=y, sr=sr), 90)

    try:
        rms = librosa.feature.rms(y=y)[0]
    except Exception as e:
        print(f"❌ ERROR RMS: {e}")
        rms = np.zeros(1)

    window_size = 20
    if len(rms) < window_size:
        window_size = max(1, len(rms) // 2)

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

        if len(segment_rms) == 0:
            avg_rms = 0
        else:
            avg_rms = np.mean(segment_rms)

        if avg_rms > high_threshold:
            intensity = "high"
        elif avg_rms > medium_threshold:
            intensity = "medium"
        else:
            intensity = "low"

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