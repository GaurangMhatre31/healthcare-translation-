import openai
from flask import Flask, request, jsonify

# 🔒 Directly Setting OpenAI API Key (Not Recommended for Production)
openai.api_key = "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"

app = Flask(__name__)

# 🎤 Speech-to-Text (Whisper API)
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    audio_file = request.files['audio']

    response = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_file,
        language="en"
    )

    return jsonify({"success": True, "text": response['text']})

# 📝 Text Translation (GPT-4)
@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    text = data.get("text")
    target_lang = data.get("target_language")

    if not text or not target_lang:
        return jsonify({"success": False, "error": "Invalid input"})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical translator."},
            {"role": "user", "content": f"Translate this medical text to {target_lang}: {text}"}
        ]
    )

    return jsonify({"success": True, "translated_text": response['choices'][0]['message']['content']})

# 🔊 Text-to-Speech (OpenAI TTS)
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"success": False, "error": "No text provided"})

    response = openai.Audio.create(
        model="tts-1",
        input=text,
        voice="alloy"
    )

    return jsonify({"success": True, "audio_url": response['url']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

