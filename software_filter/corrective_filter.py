import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fft as fft
import scipy.io.wavfile as wavfile
from scipy.constants import pi

plt.ion()

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


# Mirror response to get full spectrum
freq_start = freqs[0]
freq_end = freqs[-1]
freq_step = freqs[1] - freqs[0]
f_samp = freq_end*2
num_fill = int(freqs[0]//freq_step) #TODO ensure this is almost an integer
freqs = np.concatenate((
    np.linspace(0, freqs[0], num=num_fill, endpoint=False, dtype=freqs.dtype),
    freqs
))
freqs = np.concatenate((-freqs[::-1], freqs[1:]))
spls = np.concatenate((np.zeros(num_fill), spls))
spls = np.concatenate((spls[::-1], spls[1:]))

# Generate desired response
desired = np.zeros(len(freqs))
desired[np.abs(freqs) >= freq_start] = 75


# Plot room response
_, (room_plt, des_plt) = plt.subplots(2, sharex=True, layout="constrained")
room_plt.plot(freqs, spls)
des_plt.plot(freqs, desired)
room_plt.set_title("Room Frequency Response")
room_plt.set_ylabel("Room SPL (dB)")
des_plt.set_ylabel("Desired SPL (dB)")
des_plt.set_xlabel("$f$ (Hz)")
room_plt.grid()
des_plt.grid()


# Generate correction
correction = desired-spls

# Reduce length and smooth
ranges = 500
oddity = int(len(correction) % 2)
correction = fft.ifftshift(correction)
correction = correction[:(len(correction)+oddity)//2]
range_len = len(correction)//ranges
for i in range(ranges):
    start = range_len*i
    end = range_len*(i+1)
    correction[i] = np.mean(correction[start:end])
correction = correction[:ranges]
if oddity == 0:
    correction = np.concatenate((correction[::-1], correction))
else:
    correction = np.concatenate((correction[:0:-1], correction))
freqs = np.linspace(-f_samp/2, f_samp/2, len(correction))
correction[np.abs(freqs) <= 10] = 0

correction_linear = 10**(correction/10)
corr_filt = fft.ihfft(fft.ifftshift(correction_linear))
corr_filt = np.real(corr_filt)


# Plot correction
_, (spect_plt, imp_plt) = plt.subplots(2, layout="constrained")
spect_plt.plot(freqs, correction)
imp_plt.plot(corr_filt)
spect_plt.set_title("Correction Frequency Response")
spect_plt.set_ylabel("SPL (dB)")
spect_plt.set_xlabel("$f$ (Hz)")
spect_plt.grid()
imp_plt.set_title("Correction Impulse Response")
imp_plt.set_xlabel("$n$")
imp_plt.grid()


# Write filter to file
print("Writing filter...")
np.save("software_filter/corr_filt.npy", corr_filt)


plt.show()
input("Press ENTER to exit.")