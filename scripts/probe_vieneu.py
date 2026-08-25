"""One-off VieNeu probe: exercises the exact production synth path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from tts_vienneu import VienneuSynth

cfg = Config()
synth = VienneuSynth(cfg.vienneu_voice, backend=cfg.vienneu_backend)
pcm16, sr = synth("Xin chào các em nhỏ!")
dur = len(pcm16) / sr
peak = int(abs(pcm16).max())
assert pcm16.size > sr and peak > 1000, (
    f"suspicious output: {pcm16.size} samples, peak {peak}"
)
Path(__file__).with_name("vieneu_probe.out").write_text(
    f"voice codepoints={' '.join(f'U+{ord(c):04X}' for c in cfg.vienneu_voice)} "
    f"sr={sr} seconds={dur:.2f} peak={peak}\n",
    encoding="ascii",
)
print("probe written")
