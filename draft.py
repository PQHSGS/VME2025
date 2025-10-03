import asyncio
import io
import re
import logging
import queue
import threading
import time
from dataclasses import dataclass

import edge_tts
import numpy as np
import sounddevice as sd
from pydub import AudioSegment

# ---------- config ----------
@dataclass
class TTSConfig:
    voice: str = "vi-VN-NamMinhNeural"
    rate: str = "+0%"
    samplerate: int = 24000
    channels: int = 1
    sample_width: int = 2  # bytes (int16)
    buf_bytes: int = 32_000    # flush threshold (adjust for latency)
    flush_sec: float = 0.08    # flush tail if no incoming data
    queue_max: int = 40
    blocksize: int = 1024

# ---------- service ----------
class TTSService:
    MIN_MS = 80  # pad/fade very short clips

    def __init__(self, cfg: TTSConfig):
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
        with self._produce_lock:
            for s in sentences:
                # runs producer for one sentence, enqueues its PCM chunks before next sentence
                asyncio.run(self._download_to_queue(s))
        self._q.join()

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

# ---------- test harness ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = TTSConfig()
    svc = TTSService(cfg)
    svc.start()

    print("Type text and press Enter to speak. Type 'q' to quit.")
    try:
        while True:
            s = input("> ").strip()
            if not s:
                continue
            if s.lower() == "q":
                break
            # spawn a thread to run speak so input loop remains responsive
            threading.Thread(target=svc.speak, args=(s,), daemon=True).start()
            
    finally:
        svc.stop()
