from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg')
import os
os.environ["MPLBACKEND"] = "Agg"
import io;
import openai
from music_analysis import music_analysis
from ogg_convert import process_and_send_audio
from dotenv import load_dotenv


load_dotenv()

#ai key
openai.api_key = os.getenv("OPEN_API_KEY")

app = Flask(__name__)

global_data = {}

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024 

@app.route('/', methods=['POST'])
def index():
    global global_data
    raw_data = request.get_data()

    if len(raw_data) == 0:
        print("❌ ERROR: Data jsou prázdná")
        return "No data received", 400
    
    file_like = io.BytesIO(raw_data);
    print(file_like);

    tempo_value, length, segments, rms_tresholds = music_analysis(file_like)

    #for formating
    def format_segments(segments):
        formatted = []
        for seg in segments:
            formatted.append(f"{seg['start']:.2f} - {seg['end']:.2f} ({seg['intensity']} - {seg['rms']})")
        return formatted

    #propmpt writing, shortened, but some info might be bad, chatgpt
    def short_prompt_writing(length, tempo, segments, rms_tresholds):
        prompt = (
            "Generate rhythm-based game level data in this format: time,id,x,y,scale,direction,speed,alpha,time length. "
            "Example: 100,1,750,500,0.6,0,3,0.2,5. "
            f"length of the song: {length} "
            f"tempo of the song in bpm: {tempo} "
            f"tempo of the song in bpm: {tempo} "
            f"this is rms tresholds of the song from librosa: {rms_tresholds} "
        )
        
        for seg in segments:
            prompt += f" {seg['start']:.2f}-{seg['end']:.2f}:{seg['intensity']} (RMS={seg['rms']:.2f});"

        prompt += (
            "how to make the level: "
            "1. time is the length of the song but in this format is 1s is 17 of the value in the level time. Start on 0 and place object at least every second "
            "2. id should be random between 0 and 20 "
            "3. x and y are coordinets where the oject will spawn on the screen. They shouldn't be bigger then 1366x768 "
            "4. scale is the scale of the object adjust it based on the intensity of each segment "
            "5. direction is where the object will be going "
            "6. speed is how fast will be the object going it can be 0 "
            "7. alpha should be 0.4 "
            "8. length is how long the object will remain on the screen, it is in seconds"
            "Make sure that the level is really difficult "
            "Do not include any introductory or explanatory text; generate only the level itself. "

        )
        return prompt

    prompt = short_prompt_writing(length, tempo_value, segments, rms_tresholds)


    #ai stuff, might work when we add key idk

    try:
        gpt_response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant for game development."},
                {"role": "system", "content": "Do the whole level"},
                {"role": "user", "content": prompt}
            ]
        )
        gpt_reply = gpt_response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": f"GPT-4 API failed: {str(e)}"}), 500

    #send the stuff
    global_data['tempo_value'] = tempo_value
    global_data['segments'] = format_segments(segments)
    global_data['rms_tresholds'] = rms_tresholds
    global_data['length'] = length
    global_data['prompt'] = prompt
    global_data['reply'] = gpt_reply
    print(f"global_data after POST: {global_data}")

    return jsonify({"response": gpt_reply})

@app.route('/results', methods=['GET'])
def get_results():
    global global_data
    if 'tempo_value' not in global_data:
        return jsonify({"error": "No results available. Please process an audio file first."}), 400
    return jsonify({
        "response": global_data['reply'],
    })

if __name__ == "__main__":
    app.run()
