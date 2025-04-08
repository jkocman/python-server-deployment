from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg')
import os
os.environ["MPLBACKEND"] = "Agg"
import io
import openai
from music_analysis import music_analysis
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")

app = Flask(__name__)
CORS(app)

global_data = {}
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # max 100 MB

@app.route('/', methods=['POST'])
def index():
    global global_data
    print("📥 Přijatý POST požadavek na /")
    raw_data = request.get_data()

    if len(raw_data) == 0:
        print("⚠️ Nebyla přijata žádná data")
        return "No data received", 400

    print(f"📦 Velikost přijatých dat: {len(raw_data)} bajtů")

    file_like = io.BytesIO(raw_data)

    # Předání souboru do music_analysis
    try:
        print("▶️ Spouštím music_analysis...")
        tempo_value, length, segments, rms_tresholds = music_analysis(file_like)
        print("✅ music_analysis úspěšně dokončeno")
    except Exception as e:
        print(f"❌ Chyba v music_analysis: {e}")
        return jsonify({"error": f"Chyba při zpracování audia: {str(e)}"}), 500

    def format_segments(segments):
        return [
            f"{seg['start']:.2f} - {seg['end']:.2f} ({seg['intensity']} - {seg['rms']})"
            for seg in segments
        ]

    def short_prompt_writing(length, tempo, segments, rms_tresholds):
        print("📝 Generuju prompt pro GPT...")
        prompt = (
            "Generate rhythm-based game level data in this format: time,id,x,y,scale,direction,speed,alpha,time length. "
            "Example: 100,1,750,500,0.6,0,3,0.2,5. "
            f"length of the song: {length} "
            f"tempo of the song in bpm: {tempo} "
            f"this is rms tresholds of the song from librosa: {rms_tresholds} "
        )

        for seg in segments:
            prompt += f" {seg['start']:.2f}-{seg['end']:.2f}:{seg['intensity']} (RMS={seg['rms']:.2f});"

        prompt += (
            "how to make the level: "
            "1. time is the length of the song but in this format is 1s is 17 of the value in the level time. Start on 0 and place object at least every second "
            "2. id should be random between 0 and 20 "
            "3. x and y are coordinates where the object will spawn on the screen. They shouldn't be bigger than 1366x768 "
            "4. scale is the scale of the object adjust it based on the intensity of each segment "
            "5. direction is where the object will be going "
            "6. speed is how fast the object will be going it can be 0 "
            "7. alpha should be 0.4 "
            "8. length is how long the object will remain on the screen, it is in seconds "
            "Make sure that the level is really difficult. "
            "Do not include any introductory or explanatory text; generate only the level itself. "
        )
        print("✅ Prompt hotový")
        return prompt

    prompt = short_prompt_writing(length, tempo_value, segments, rms_tresholds)

    try:
        print("🤖 Posílám požadavek na GPT...")
        gpt_response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant for game development."},
                {"role": "user", "content": prompt}
            ]
        )
        gpt_reply = gpt_response.choices[0].message.content
        print("✅ GPT odpověď získána")
    except Exception as e:
        print(f"❌ Chyba při volání GPT API: {e}")
        return jsonify({"error": f"GPT-4 API failed: {str(e)}"}), 500

    global_data = {
        "tempo_value": tempo_value,
        "segments": format_segments(segments),
        "rms_tresholds": rms_tresholds,
        "length": length,
        "prompt": prompt,
        "reply": gpt_reply
    }

    print("✅ Výstup vrácen klientovi")
    return jsonify({"response": gpt_reply})

@app.route('/results', methods=['GET'])
def get_results():
    global global_data
    print("📥 GET požadavek na /results")
    if 'tempo_value' not in global_data:
        print("⚠️ Není k dispozici žádný výsledek")
        return jsonify({"error": "No results available. Please process an audio file first."}), 400
    print("✅ Výsledky nalezeny")
    return jsonify({"response": global_data['reply']})

if __name__ == "__main__":
    print("🚀 Spouštím Flask server...")
    app.run()
