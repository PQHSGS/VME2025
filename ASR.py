from flask import Flask, request, jsonify
import numpy as np
import logging
import soundfile as sf
import io
from transformers import pipeline
from faster_whisper import WhisperModel
import google.generativeai as genai

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Global ASR model instance
asr_model = None
model_path = "./ct2-phowhisper-small"
def load_asr_model():
    """Loads the ASR model once when the application starts."""
    global asr_model
    try:
        logging.info(f"Loading ASR model '{model_path}'...")
        asr_model = WhisperModel(model_path, device="cpu", compute_type="int8")
        logging.info("ASR model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load ASR model: {e}")
        asr_model = None

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Endpoint to receive an audio file and return its transcription."""
    if asr_model is None:
        return jsonify({'error': 'ASR model not loaded'}), 503

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if not audio_file.filename.endswith(('.wav', '.mp3', '.flac')):
        return jsonify({'error': 'Invalid file format. Please use .wav, .mp3, or .flac'}), 415

    try:
        # Read the audio file into a byte stream
        audio_bytes = io.BytesIO(audio_file.read())
        
        # Load audio data using soundfile
        audio_data, sr = sf.read(audio_bytes, dtype='float32')
        
        if len(audio_data.shape) > 1:
            # Convert stereo to mono by averaging channels
            audio_data = np.mean(audio_data, axis=1)

        # Transcribe the audio
        logging.info("Transcribing audio...")
        transcription_result, _ = asr_model.transcribe(
            audio_data,
            beam_size=1,                   # explore a few alternatives for accuracy
            temperature=0.0,                 # still deterministic
            no_speech_threshold=0.7,        # a bit stricter than default
            hallucination_silence_threshold=1.5,  # tolerate longer pauses
            log_prob_threshold=-1.0          # allow more low-confidence words
        )


        text = " ".join([segment.text for segment in transcription_result])

        logging.info(f"Transcription complete: {text}")
        return jsonify({'transcription': text})

    except Exception as e:
        logging.error(f"Transcription process failed: {e}")
        return jsonify({'error': 'Transcription failed'}), 500

if __name__ == '__main__':
    load_asr_model()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)