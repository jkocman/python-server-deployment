import librosa
import numpy as np

def music_analysis(file_like):
    print("🔍 Začínám načítat audio...")
    try:
        y, sr = librosa.load(file_like, duration=30.0)
        print(f"✅ Načteno: {len(y)} vzorků, vzorkovací frekvence: {sr}")
    except Exception as e:
        print(f"❌ Chyba při načítání audia: {e}")
        return "Error loading audio", 500

    try:
        print("🔍 Detekce tempa...")
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_value = float(tempo)
        print(f"✅ Detekováno tempo: {tempo_value}")
    except Exception as e:
        print(f"❌ Chyba při detekci tempa: {e}")
        return "Error detecting tempo", 500

    try:
        print("🔍 Výpočet délky skladby...")
        length = librosa.get_duration(y=y, sr=sr)
        if length >= 90:
            length = 90
        print(f"✅ Délka skladby: {length} s")
    except Exception as e:
        print(f"❌ Chyba při výpočtu délky: {e}")
        return "Error getting duration", 500

    try:
        print("🔍 Výpočet RMS...")
        rms = librosa.feature.rms(y=y)[0]
        print(f"✅ RMS spočítáno: {len(rms)} hodnot")
        window_size = 50
        rms_smooth = np.convolve(rms, np.ones(window_size) / window_size, mode='same')
        print(f"✅ RMS vyhlazeno")
    except Exception as e:
        print(f"❌ Chyba při výpočtu RMS: {e}")
        return "Error computing RMS", 500

    try:
        print("🔍 Výpočet prahů intenzity...")
        median_rms = np.median(rms_smooth)
        low_threshold = median_rms * 0.8
        medium_threshold = median_rms * 1.0
        high_threshold = median_rms * 1.2
        print(f"✅ Prahy: low={low_threshold}, medium={medium_threshold}, high={high_threshold}")
    except Exception as e:
        print(f"❌ Chyba při výpočtu prahů intenzity: {e}")
        return "Error computing thresholds", 500

    try:
        print("🔍 Segmentace skladby...")
        segment_duration = 10
        total_duration = len(y) / sr
        num_segments = int(np.ceil(total_duration / segment_duration))
        segments = []
        print(f"✅ Celková délka: {total_duration}, segmentů: {num_segments}")

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
        print(f"✅ Segmentace dokončena: {len(segments)} segmentů")
    except Exception as e:
        print(f"❌ Chyba při segmentaci: {e}")
        return "Error segmenting audio", 500

    rms_tresholds = [
        f"high_threshold {high_threshold}",
        f"medium_threshold {medium_threshold}",
        f"low_threshold {low_threshold}"
    ]

    print("✅ Všechno hotovo")
    return tempo_value, length, segments, rms_tresholds
