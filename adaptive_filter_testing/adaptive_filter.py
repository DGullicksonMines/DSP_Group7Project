import math
import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.signal as sig
import scipy.fft as fft
import scipy.io.wavfile as wavfile
from scipy.constants import pi

import sounddevice as sd
# Documentation: https://python-sounddevice.readthedocs.io/en/0.4.6/usage.html

# Open audio file
f_samp, audio = wavfile.read("Audio/Radioactive.wav")

# # Combine channels by averaging
# audio = (audio[:, 0] + audio[:, 1]) / 2
# Use only one channel
audio = audio[:, 0]

# Split audio file into parts
PART_DURATION_MS = 2000 # Duration of each part in ms
part_samples = f_samp * PART_DURATION_MS // 1000 # Duration of each part in samples
num_parts = math.ceil(len(audio) / part_samples) # Number of parts
parts = []
for i in range(num_parts):
    start = part_samples*i
    end = part_samples*(i+1)
    parts.append(audio[start:end])

# For each part:
#   Play part while recording from microphone
#   Compare recorded spectrum to expected spectrum
#   Update compensating filter for next part

# Prepare plot
freqs = np.linspace(-f_samp/2, f_samp/2, part_samples, endpoint=False)
plt.ion()
fig = plt.figure()
exp_line: Line2D = plt.plot(freqs, np.zeros(part_samples), label="Expected")[0]
obs_line: Line2D = plt.plot(freqs, np.zeros(part_samples), label="Observed")[0]
plt.title("Frequency Spectrum")
plt.xlabel(r"$f$")
plt.legend()
plt.draw()
plt.pause(0.001)

# Iterate over parts
last_part = None
last_attenuation = None
for i in range(len(parts)):
    part = parts[i]

    # Play part AND record
    recording = sd.playrec(part, f_samp, channels=1)

    # Wait for completion
    sd.wait()

    # # Playback recording
    # sd.play(recording)
    # sd.wait()

    # Get expected spectrum
    exp_spect = fft.fftshift(fft.fft(part))
    # Get observed spectrum
    obs_spect = fft.fftshift(fft.fft(recording))

    exp_spect *= np.max(obs_spect)/np.max(exp_spect)

    # Replot observed and expected spectrums
    # Update data directly for speed
    exp_line.set_ydata(np.abs(exp_spect))
    obs_line.set_ydata(np.abs(obs_spect))

    # Allow plot to update
    fig.axes[0].relim()
    fig.axes[0].autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)

time.sleep(5)