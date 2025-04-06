import librosa
import soundfile as sf
from flask import send_file

def convert_to_ogg(input_path, output_path="converted_audio.ogg"):
    data, samplerate = librosa.load(input_path, sr=None)  # Načte libovolný zvuk
    sf.write(output_path, data, samplerate, format='OGG', subtype='VORBIS')  # Uloží do OGG
    return output_path

def process_and_send_audio(raw_data):
    """Uloží soubor, převede ho na OGG a pošle zpět"""
    if not raw_data:
        return "No data received", 400

    input_path = "input_audio.mp3"
    try:
        with open(input_path, "wb") as f:
            f.write(raw_data)

        ogg_path = convert_to_ogg(input_path)
        if not ogg_path:
            return "Error during conversion", 500

        return send_file(ogg_path, mimetype="audio/ogg", as_attachment=True)
    except Exception as e:
        return f"Internal server error: {e}", 500
