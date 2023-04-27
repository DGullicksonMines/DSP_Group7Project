import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fft as fft
import scipy.io.wavfile as wavfile
from scipy.constants import pi

# Documentation: https://python-sounddevice.readthedocs.io/en/0.4.6/usage.html
import sounddevice as sd

import calibration as c

# Get audio
f_samp, audio = wavfile.read("adaptive_filter_testing/Audio/Radioactive.wav")
audio = audio[:, 0]

# Apply filter
print("Applying filter to audio...")
# filtered = sig.filtfilt(b=c.calib_filt, a=1, x=audio)
filtered = sig.convolve(c.calib_filt, audio)

# Play audio
print("Playing audio!")
sd.play(filtered, samplerate=f_samp, blocking=True)