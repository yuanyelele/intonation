import math
import os
import random
import sys
import tty
import librosa
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pyloudnorm as pyln
import scipy
import sounddevice as sd

SAMPLE_RATE = 48000
METER = pyln.Meter(SAMPLE_RATE)
LOUDNESS = -18  # -65 barely audible (master -12db)
FMIN = librosa.note_to_hz("A1")  # A0 is lowest key on piano
FMAX = librosa.note_to_hz("C7")  # C8 is highest key on piano
UP = b"\x1b[A"
DOWN = b"\x1b[B"
LEFT = b"\x1b[D"
ESC = b"\x1b"
STEP = 0.9  # ∈ (0, 1)
DIFF = 20
STATS_FILE = "stats.npy"

sd.default.device = "system"
sd.default.channels = 2
sd.default.samplerate = SAMPLE_RATE


def fade(y, length=1024):
    y[:length] *= np.linspace(0, 1, length)
    y[-length:] *= np.linspace(1, 0, length)


def play(waveform="sine", f=440, dur=1, pan=0, meter=METER):
    x = np.arange(SAMPLE_RATE * dur) * 2 * math.pi * f / SAMPLE_RATE
    match waveform:
        case "square":
            y = scipy.signal.square(x)
        case "sawtooth":
            y = scipy.signal.sawtooth(x)
        case "triangle":
            y = scipy.signal.sawtooth(x, 0.5)
        case "sine":
            y = np.sin(x)
        case _:
            print("unknown waveform type")
            return
    fade(y)
    loudness = meter.integrated_loudness(y)
    y = pyln.normalize.loudness(y, loudness, LOUDNESS)
    y = np.column_stack((y * (1 - pan) / 2, y * (1 + pan) / 2))
    sd.play(y, mapping=[1, 2])
    sd.wait()


def gen_tones(level):
    diff = DIFF * STEP**level
    f1 = math.exp(random.uniform(math.log(FMIN), math.log(FMAX)))
    direction = 1 if random.random() < 0.5 else -1
    f2 = f1 * 2 ** (diff / 1200 * direction)
    print(f"{diff:.2f}", end=" ", flush=True)
    return f1, f2, direction


def get_key():
    while True:
        b = os.read(sys.stdin.fileno(), 3)
        if b == ESC:
            return "esc"
        if b == UP:
            return "up"
        if b == DOWN:
            return "down"
        if b == LEFT:
            return "left"


def report(stats):
    levels, freqs, colors = np.array(stats).T
    diff = DIFF * STEP**levels
    _, ax = plt.subplots()
    ax.loglog()
    for axis in (ax.xaxis, ax.yaxis):
        formatter = FuncFormatter(lambda y, _: f"{y:g}")
        axis.set_major_formatter(formatter)
        axis.set_minor_formatter(formatter)
    ax.set_xlabel("diff (cents)")
    ax.set_ylabel("freq (hz)")
    ax.scatter(diff, freqs, c=colors)
    plt.show()


def main():
    stats = list(np.load(STATS_FILE)) if os.path.exists(STATS_FILE) else []
    level = 0
    tty.setcbreak(sys.stdin.fileno())
    f1, f2, direction = gen_tones(level)
    while True:
        play("triangle", f1)
        play("triangle", f2)
        key = get_key()
        if key == "esc":
            break
        if key == "left":
            continue
        ans = 1 if key == "up" else -1
        if ans == direction:
            print("✓")
            stats.append((level, (f1 * f2) ** 0.5, True))
            level += 1
        else:
            print("✗")
            stats.append((level, (f1 * f2) ** 0.5, False))
            level -= 2
        f1, f2, direction = gen_tones(level)
    print("stopping")
    os.system("stty sane")
    report(stats)
    np.save(STATS_FILE, stats)


if __name__ == "__main__":
    main()
