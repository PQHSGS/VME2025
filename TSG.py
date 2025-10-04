from dataclasses import dataclass, field
from datetime import datetime
import sounddevice as sd
import numpy as np
import threading
import requests
import asyncio
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
import os

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
    rag_url: str = "http://127.0.0.1:7000/chat"
    webhook_timeout: int = 12
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    # TTS
    voice: str = "vi-VN-NamMinhNeural"
    rate: str = "+10%"
    samplerate: int = 24000
    channels: int = 1
    sample_width: int = 2  # bytes (int16)
    buf_bytes: int = 32_000    # flush threshold (adjust for latency)
    flush_sec: float = 0.08    # flush tail if no incoming data
    queue_max: int = 40
    blocksize: int = 1024

    # Fallback
    user_backup_str: str = "<system> Đang có vấn đề về đường truyền"
    model_backup_str: str = "Cháu chờ ông một chút nha."

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
        filename="tsg.log"
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

        if audio_data.size == 0:
            self.logger.warning("No audio data was recorded.")
            return self.cfg.backup_str
        record_id = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("Records", exist_ok=True)
        sf.write(f'Records/record_{record_id}.wav', audio_data, self.cfg.sample_rate)
        self.logger.info("Sending to ASR server...")
        transcribed_text = self.stt.transcribe_audio(audio_data)
        return transcribed_text if transcribed_text else self.cfg.user_backup_str

# -----------------------------
# RAG Client (currently posts to Gemini)
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
    MIN_MS = 80  # pad/fade very short clips

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger("TTSService")
        self._q = queue.Queue(maxsize=cfg.queue_max)
        self._play_thread = None
        self._play_thread_stop = threading.Event()
        self._produce_lock = threading.Lock()

    # decode full mp3 bytes -> int16 numpy PCM
    def _decode_mp3_bytes_to_pcm(self, buf_bytes: bytes):
        audio = AudioSegment.from_file(io.BytesIO(buf_bytes), format="mp3")
        audio = audio.set_frame_rate(self.cfg.samplerate) \
                     .set_channels(self.cfg.channels) \
                     .set_sample_width(self.cfg.sample_width)
        if len(audio) < self.MIN_MS:
            audio = audio.fade_in(int(self.MIN_MS/4)).fade_out(int(self.MIN_MS/4))
            if len(audio) < self.MIN_MS:
                audio = audio + AudioSegment.silent(duration=self.MIN_MS - len(audio))
        raw = audio.raw_data
        pcm = np.frombuffer(raw, dtype=np.int16)
        if self.cfg.channels > 1:
            pcm = pcm.reshape(-1, self.cfg.channels)
        return pcm

    # producer: stream edge-tts mp3 frames and decode then put PCM into queue
    async def _download_to_queue(self, text: str):
        comm = edge_tts.Communicate(text, voice=self.cfg.voice, rate=self.cfg.rate)
        buf = bytearray()
        last = time.time()
        try:
            async for chunk in comm.stream():
                if chunk.get("type") == "audio":
                    buf.extend(chunk.get("data", b""))
                    last = time.time()
                # flush on size
                if buf and len(buf) >= self.cfg.buf_bytes:
                    try:
                        pcm = self._decode_mp3_bytes_to_pcm(bytes(buf))
                        self._q.put(pcm)
                    except Exception:
                        self.logger.exception("decode error on chunked buffer; dropping")
                    buf.clear()
                # flush small tail when stream stalls briefly
                if buf and (time.time() - last) > self.cfg.flush_sec:
                    try:
                        pcm = self._decode_mp3_bytes_to_pcm(bytes(buf))
                        self._q.put(pcm)
                    except Exception:
                        self.logger.exception("decode error on flush buffer; dropping")
                    buf.clear()
        except Exception:
            self.logger.exception("edge_tts stream error")
        # final leftover
        if buf:
            try:
                pcm = self._decode_mp3_bytes_to_pcm(bytes(buf))
                self._q.put(pcm)
            except Exception:
                self.logger.exception("final decode error; dropping")

    # consumer: playback thread that writes PCM to OutputStream
    def _playback_worker(self):
        try:
            with sd.OutputStream(samplerate=self.cfg.samplerate,
                                 channels=self.cfg.channels,
                                 dtype="int16",
                                 blocksize=self.cfg.blocksize) as out:
                while not self._play_thread_stop.is_set():
                    item = self._q.get()
                    if item is None:
                        # sentinel to stop
                        break
                    try:
                        pcm = item  # numpy int16 array
                        out.write(pcm)
                    except Exception:
                        self.logger.exception("playback write error")
                    finally:
                        self._q.task_done()
        except Exception:
            self.logger.exception("output stream error")

    # start playback thread once
    def start(self):
        if self._play_thread is None or not self._play_thread.is_alive():
            self._play_thread_stop.clear()
            self._play_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self._play_thread.start()
            self.logger.debug("Playback thread started")
    # split text into sentences/phrases of max_len (approx) for better TTS pacing
    def _split_sentences(self, text, max_len=180):
        parts = re.split(r'(?<=[.!?])\s+', text)
        merged, buf = [], ""
        for p in parts:
            if len(buf) + len(p) < max_len:
                buf += " " + p
            else:
                if buf:
                    merged.append(buf.strip())
                buf = p
        if buf:
            merged.append(buf.strip())
        return merged

    # speak a single text (blocks until production finishes, playback runs in background)
    def speak(self, text: str):
        if not text:
            self.logger.debug("Empty text for TTS.speak")
            return
        self.logger.debug("Starting TTS download for text length=%d", len(text))
        
        sentences = self._split_sentences(text, max_len=150)
       # add something
        with self._produce_lock:
            for s in sentences:
                # runs producer for one sentence, enqueues its PCM chunks before next sentence
                asyncio.run(self._download_to_queue(s))
        #end sth 

    # stop service gracefully
    def stop(self):
        # signal playback worker to stop after queue drains
        self._q.put(None)
        self._play_thread_stop.set()
        if self._play_thread:
            self._play_thread.join(timeout=2.0)
        # clear queue
        with self._q.mutex:
            self._q.queue.clear()
        self.logger.debug("TTSService stopped")

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
        self.tts.start()
        # session
        self.session_id = make_request_id(cfg.id_prefix)

    def run(self):
        self.logger.info("Conversation manager started. Press ENTER to record, 'q' to quit.")
        try:
            while True:
                cmd = input("Press ENTER to record, or type 'q' to quit: ").strip().lower()
                print(f"start listening...")
                if cmd == "q":
                    self.logger.info("Quitting conversation loop")
                    break

                transcribed_text = self.recorder.record_and_transcribe().strip()
                print(f"You said: {transcribed_text}")
                #transcribed_text = input("Enter your message: ")
                self.logger.info(f"User: {transcribed_text}")

                rag_reply = self.rag.send_to_rag(transcribed_text, session_id=self.session_id)
                print(f"RAG reply: {rag_reply}")
                if not rag_reply:
                    self.logger.warning("Empty reply from RAG; using backup string")
                    rag_reply = self.cfg.model_backup_str

                self.logger.info(f"Bot: {rag_reply}")
                threading.Thread(target=self.tts.speak, args=(rag_reply,), daemon=True).start()

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == '__main__':
    cfg = Config()
    manager = ConversationManager(cfg, verbose=True)
    manager.run()