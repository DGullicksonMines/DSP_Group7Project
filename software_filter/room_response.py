import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fft as fft
import scipy.io.wavfile as wavfile
from scipy.constants import pi

# Read text file into np array
freqs = []
spls = []
with open("software_filter/AvgFreqResponse_Better.txt") as file:
    # Ignore first 14 lines
    for _ in range(14): _ = file.readline()

    # Fill arrays
    while line := file.readline():
        freq, spl = line.split(sep=None)
        freqs.append(float(freq))
        spls.append(float(spl))
freqs = np.array(freqs)
spls = np.array(spls)


# Plot room response
plt.figure()
plt.plot(freqs, spls)
plt.title("Room Frequency Response")
plt.ylabel("Room SPL (dB)")
plt.xlabel("$f$ (Hz)")
plt.grid()
plt.show()