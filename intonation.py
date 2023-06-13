import argparse
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
LOUDNESS = -18
CALIB_LOUDNESS = -65
FMIN = librosa.note_to_hz("A1")  # A0 is lowest key on piano
FMAX = librosa.note_to_hz("C7")  # C8 is highest key on piano
UP = b"\x1b[A"
DOWN = b"\x1b[B"
LEFT = b"\x1b[D"
ESC = b"\x1b"
STEP = 0.9  # ∈ (0, 1)
DIFF = 20

sd.default.device = "system"
sd.default.channels = 2
sd.default.samplerate = SAMPLE_RATE
try:
    sd.check_output_settings()
except sd.PortAudioError:
    # see sounddevice PR #85
    print("invalid sounddevice settings, try different device/channels/samplerate")
    sys.exit()()


def fade(y, length=1024):
    y[:length] *= np.linspace(0, 1, length)
    y[-length:] *= np.linspace(1, 0, length)


def play(waveform="sine", f=440, pan=0, dur=1, loudness=LOUDNESS):
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
    y = pyln.normalize.loudness(y, METER.integrated_loudness(y), loudness)
    y = np.column_stack((y * (1 - pan) / 2, y * (1 + pan) / 2))
    sd.play(y, mapping=[1, 2])
    sd.wait()


def calibrate(args):
    print("adjust computer volume until the notes are barely audible")
    print("ctrl-c to end the loop when done")
    while True:
        f = math.exp(random.uniform(math.log(FMIN), math.log(FMAX)))
        play(args.waveform, f, pan=args.pan, loudness=CALIB_LOUDNESS)


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
    parser = argparse.ArgumentParser(
        description="intonation ear training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--calib",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="do calibration",
    )
    parser.add_argument(
        "--pan",
        type=float,
        default=0,
        help="panning (-1 for left, 1 for right, 0 for centre)",
    )
    parser.add_argument(
        "--waveform",
        choices=("square", "sawtooth", "triangle", "sine"),
        default="triangle",
        help="type of waveform",
    )
    parser.add_argument(
        "--stats",
        default="stats.npy",
        help="filename to save/append statistics, or 'none'",
    )
    args = parser.parse_args()

    if args.calib:
        calibrate(args)
        sys.exit()

    print("Press 'up' if the second tone is higher, or 'down' if it is lower.")
    print("Press 'left' to listen to the pair again.")
    print("Press 'esc' to quit and view the report.")

    stats = []
    if args.stats != "none" and os.path.exists(args.stats):
        stats = list(np.load(args.stats))
    tty.setcbreak(sys.stdin.fileno())

    level = 0
    f1, f2, direction = gen_tones(level)
    while True:
        play(args.waveform, f1, pan=args.pan)
        play(args.waveform, f2, pan=args.pan)
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
    np.save(args.stats, stats)


if __name__ == "__main__":
    main()
