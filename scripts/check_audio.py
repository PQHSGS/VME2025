"""Audio device sanity check for the kiosk box.

Verifies mic input + speaker output exist and that the exact stream
parameters the app uses (16 kHz float32 mono in, 24 kHz int16 mono out)
open cleanly on the WASAPI defaults. Run before every show:

    python scripts/check_audio.py
"""

from __future__ import annotations

import sys
import time

import numpy as np


def main() -> int:
    import sounddevice as sd

    print(f"sounddevice {sd.__version__} | PortAudio {sd.get_portaudio_version()[1]}")
    print(f"default input : {sd.default.device[0]}")
    print(f"default output: {sd.default.device[1]}")

    devices = sd.query_devices()
    inputs = [
        (i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0
    ]
    outputs = [
        (i, d) for i, d in enumerate(devices) if d["max_output_channels"] > 0
    ]
    print(f"\n{len(inputs)} input device(s):")
    for i, d in inputs:
        print(
            f"  [{i}] {d['name'][:50]:<50} in_ch={d['max_input_channels']} "
            f"default_sr={d['default_samplerate']}"
        )
    print(f"{len(outputs)} output device(s):")
    for i, d in outputs:
        print(
            f"  [{i}] {d['name'][:50]:<50} out_ch={d['max_output_channels']} "
            f"default_sr={d['default_samplerate']}"
        )

    ok = True

    # --- capture path: 16 kHz float32 mono, 30ms blocks (audio.py params) --
    try:
        with sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="float32",
            blocksize=480,
        ) as stream:
            frame, _ = stream.read(480)
            rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))
            time.sleep(0.2)
        print(f"\n[capture] 16kHz mono OK (rms of first frame={rms:.5f})")
        ok &= True
    except Exception as exc:
        print(f"\n[capture] FAIL: {exc}")
        ok = False

    # --- playback path: 24 kHz int16 mono, ~200ms writes (tts.py params) ---
    # Writes SILENCE only - verifies the device accepts our format/rate.
    try:
        silence = np.zeros(4800, dtype=np.int16)
        with sd.OutputStream(
            samplerate=24000, channels=1, dtype="int16"
        ) as stream:
            stream.write(silence)  # blocking, ~200ms of nothing
        print("[playback] 24kHz int16 OK (wrote 200ms silence)")
    except Exception as exc:
        print(f"[playback] FAIL: {exc}")
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
