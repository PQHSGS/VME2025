from dataclasses import dataclass, field
from datetime import datetime
import sounddevice as sd
import numpy as np
import threading
import requests
import asyncio
from transformers import pipeline
import edge_tts
import io
import soundfile as sf
import time
import re
import uuid
import logging
from typing import List, Optional
import wave
from pydub import AudioSegment
import queue


# -----------------------------
# Configuration / Hyperparams
# -----------------------------

@dataclass
class Config:
    # Audio / recording
    sample_rate: int = 16000
    channels: int = 1
    record_duration_seconds: float = 30.0

    #ASR
    asr_url: str = 'http://127.0.0.1:5000/transcribe'
    # RAG / webhook
    rag_url: str = "http://127.0.0.1:8000/chat"
    webhook_timeout: int = 30
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    # TTS
    voice: str = "vi-VN-NamMinhNeural"
    samplerate = 24000
    channels = 1
    sample_width = 2  # bytes -> 16-bit
    # tuning
    buf_bytes = 50_000   # larger -> smoother, more latency
    flush_sec = 0.12     # flush if no new chunk for this time
    rate: str = "+10%"
    play_thread = None

    # Fallback
    backup_str: str = "ông có biết sự tích chú cuội cung trăng không ạ"

    # Language
    language: str = "vi"

    # Misc
    id_prefix: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3])


# -----------------------------
# Verbose / Logging Setup
# -----------------------------

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=level,
    )


# -----------------------------
# Utilities
# -----------------------------

def make_request_id(prefix: str = "") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# -----------------------------
# STT Service (Whisper)
# -----------------------------
class WhisperSTT:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("WhisperSTT client initialized. Calling localhost ASR service.")

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        if audio_data.size == 0:
            self.logger.debug("Empty audio array given to transcribe_audio.")
            return ""

        # Convert numpy array to an in-memory WAV file
        in_memory_wav = io.BytesIO()
        with wave.open(in_memory_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # For 16-bit audio
            wf.setframerate(self.cfg.sample_rate)
            audio_data_int16 = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_data_int16.tobytes())
        in_memory_wav.seek(0)
        
        files = {'audio': ('audio.wav', in_memory_wav, 'audio/wav')}
        asr_endpoint = self.cfg.asr_url

        try:
            self.logger.info(f"Sending audio to ASR server at {asr_endpoint}")
            response = requests.post(asr_endpoint, files=files, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            transcribed_text = result.get('transcription', '').strip()
            
            if transcribed_text:
                self.logger.info(f"Transcription received: {transcribed_text}")
            else:
                self.logger.warning("Server returned an empty transcription.")
            
            return transcribed_text
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to connect to ASR server: {e}")
            return ""
        except Exception as e:
            self.logger.exception("An unexpected error occurred during transcription.")
            return ""


# -----------------------------
# Recorder (handles streaming input from mic)
# -----------------------------
import sounddevice as sd
import numpy as np
import threading
import logging
import time

class Recorder:
    def __init__(self, cfg: Config, stt: WhisperSTT):
        self.cfg = cfg
        self.stt = stt
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop_flag = threading.Event()

    def record_and_transcribe(self) -> str:
        self.logger.info(
            f"Recording (max {self.cfg.record_duration_seconds} seconds)... Press ENTER to stop early."
        )

        # Buffer for recorded audio
        duration_samples = int(self.cfg.record_duration_seconds * self.cfg.sample_rate)
        audio_data = np.zeros((duration_samples, self.cfg.channels), dtype="float32")

        # Launch recorder in background
        def _rec():
            try:
                sd.rec(
                    out=audio_data,
                    samplerate=self.cfg.sample_rate,
                    channels=self.cfg.channels,
                    dtype="float32",
                )
                sd.wait()
            except Exception:
                self.logger.exception("Recording error")

        rec_thread = threading.Thread(target=_rec, daemon=True)
        rec_thread.start()

        # Wait for ENTER or duration
        start = time.time()
        try:
            input()  # blocks until ENTER
            self._stop_flag.set()
            sd.stop()
            self.logger.info("Recording stopped early by ENTER.")
        except Exception:
            pass

        rec_thread.join()
        elapsed = time.time() - start
        self.logger.info(f"Recording finished (elapsed {elapsed:.1f}s).")

        # Trim silence if stopped early
        max_frames = int(elapsed * self.cfg.sample_rate)
        if max_frames < len(audio_data):
            audio_data = audio_data[:max_frames]

        if audio_data.size == 0:
            self.logger.warning("No audio data was recorded.")
            return self.cfg.backup_str

        self.logger.info("Sending to ASR server...")
        transcribed_text = self.stt.transcribe_audio(audio_data)
        return transcribed_text if transcribed_text else self.cfg.backup_str

# -----------------------------
# RAG Client (currently posts to webhook/Gemini)
# -----------------------------
class RAGClient:
    """
    Currently implemented as a webhook client that sends the transcript to an RAG endpoint
    that uses Gemini as generator.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_response(self, response: requests.Response) -> str:
        try:
                # Extract raw text
            text = response.json()['text']
            # Replace literal escape sequences (\n, \t) with actual whitespace
            text = text.replace("\\n", "\n").replace("\\t", " ")
            # Turn double newlines into sentence break
            text = re.sub(r'\n\s*\n+', ' ', text)
            # Turn single newlines into space
            text = re.sub(r'[\n\r]+', ' ', text)
            # Remove unwanted symbols but KEEP Vietnamese letters
            text = re.sub(r'[/\\"*]+', ' ', text)
            # Remove URLs
            text = re.sub(r'http\S+|www\.\S+', '', text)
            # Remove code-like brackets
            text = re.sub(r'[{<>}]+', ' ', text)
            # Normalize spaces around numbers
            text = re.sub(r'(\d+)', r' \1 ', text)
            # Collapse multiple spaces
            text = re.sub(r'\s+', ' ', text)
            # Trim
            text = text.strip()
            # Capitalize first letter
            if text:
                text = text[0].upper() + text[1:]
            return text
        except Exception:
            self.logger.exception("Failed to parse RAG response")
            return ""

    def send_to_rag(self, transcript: str, session_id: Optional[str] = None) -> str:
        session_id = session_id or make_request_id(self.cfg.id_prefix)
        payload = {
            "id": session_id,
            "data": transcript,
        }
        attempt = 0
        while attempt < self.cfg.max_retries:
            try:
                self.logger.debug(f"Sending to RAG webhook (attempt {attempt+1}): {self.cfg.rag_url}")
                r = requests.post(self.cfg.rag_url, json=payload, timeout=self.cfg.webhook_timeout)
                r.raise_for_status()
                ans = self.process_response(r)
                self.logger.info(f"RAG reply (len={len(ans)})")
                return ans
            except Exception as e:
                attempt += 1
                self.logger.warning(f"RAG request failed (attempt {attempt}). Error: {e}")
                time.sleep(self.cfg.retry_backoff_seconds * attempt)
        self.logger.error("RAG request failed after retries")
        return ""


# -----------------------------
# TTS Service (edge-tts)
# -----------------------------
class TTSService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        self._q = queue.Queue(maxsize=20)

    async def _download_to_queue(self, text: str):
        comm = edge_tts.Communicate(text, voice=self.cfg.voice, rate=self.cfg.rate)
        buf = bytearray()
        last = time.time()
        try:
            async for chunk in comm.stream():
                if chunk.get("type") == "audio":
                    buf.extend(chunk.get("data", b""))
                    last = time.time()
                    if len(buf) >= self.cfg.buf_bytes:
                        self._q.put(bytes(buf))
                        buf.clear()
                # flush small tail when stream stalls briefly
                if buf and (time.time() - last) > self.cfg.flush_sec:
                    self._q.put(bytes(buf))
                    buf.clear()
        except Exception:
            self.logger.exception("edge_tts stream error")
        if buf:
            self._q.put(bytes(buf))

    def _playback_worker(self):
        try:
            with sd.OutputStream(samplerate=self.cfg.samplerate,
                                 channels=self.cfg.channels,
                                 dtype="int16",
                                 blocksize=0) as out:
                while True:
                    item = self._q.get()
                    if item is None:
                        break
                    try:
                        audio = AudioSegment.from_file(io.BytesIO(item), format="mp3")
                        audio = audio.set_frame_rate(self.cfg.samplerate)\
                                     .set_channels(self.cfg.channels)\
                                     .set_sample_width(self.cfg.sample_width)
                        pcm = np.frombuffer(audio.raw_data, dtype=np.int16)
                        if self.cfg.channels > 1:
                            pcm = pcm.reshape(-1, self.cfg.channels)
                        out.write(pcm)
                    except Exception:
                        self.logger.exception("playback decode/write error")
        except Exception:
            self.logger.exception("output stream error")

    def speak(self, text: str):
        if not text:
            self.logger.debug("Empty text for TTS.speak")
            return
        self.logger.debug("Starting TTS stream...")
        self.cfg.play_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.cfg.play_thread.start()
        try:
            asyncio.run(self._download_to_queue(text))
        except Exception:
            self.logger.exception("TTS download failed")
        finally:
            self._q.put(None)
            self.cfg.play_thread.join()
            self.logger.debug("TTS stream finished")


# -----------------------------
# Conversation Manager
# -----------------------------
class ConversationManager:
    def __init__(self, cfg: Config, verbose: bool = False):
        setup_logging(verbose)
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        # instantiate components
        self.stt = WhisperSTT(cfg)
        self.recorder = Recorder(cfg, self.stt)
        self.rag = RAGClient(cfg)
        self.tts = TTSService(cfg)
        self.session_id = make_request_id(cfg.id_prefix)

    def run(self):
        self.logger.info("Conversation manager started. Press ENTER to record, 'q' to quit.")
        try:
            while True:
                cmd = input("Press ENTER to record, or type 'q' to quit: ").strip().lower()
                if cmd == "q":
                    self.logger.info("Quitting conversation loop")
                    break

                transcribed_text = self.recorder.record_and_transcribe().strip()
                self.logger.info(f"User: {transcribed_text}")

                rag_reply = self.rag.send_to_rag(transcribed_text, session_id=self.session_id)
                if not rag_reply:
                    self.logger.warning("Empty reply from RAG; using backup string")
                    rag_reply = self.cfg.backup_str

                self.logger.info(f"Bot: {rag_reply}")
                self.tts.speak(rag_reply)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == '__main__':
    cfg = Config()
    manager = ConversationManager(cfg, verbose=True)
    manager.run()