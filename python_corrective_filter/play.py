import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fft as fft
import scipy.io.wavfile as wavfile
from scipy.constants import pi

# Documentation: https://python-sounddevice.readthedocs.io/en/0.4.6/usage.html
import sounddevice as sd

plt.ion()

# Get audio
f_samp, audio = wavfile.read("Audio.wav")
audio = audio[:, 0]
original_type = audio.dtype

# Apply filter
corr_filt = np.load("corr_filt.npy")
print("Applying filter to audio...")
filtered = sig.convolve(audio, corr_filt, mode="same")

# Plot spectrums
freqs = np.linspace(-f_samp/2, f_samp/2, len(audio))
freqs_filt = np.linspace(-f_samp/2, f_samp/2, len(corr_filt))
orig_spect = fft.fftshift(fft.hfft(audio, len(audio)))
filt_spect = fft.fftshift(fft.hfft(corr_filt, len(corr_filt)))
filtrd_spect = fft.fftshift(fft.hfft(filtered, len(audio)))
_, (orig_plt, filt_plt, filtrd_plt) = plt.subplots(3, sharex=True, layout="constrained")
orig_plt.plot(freqs, 10*np.log10(np.abs(orig_spect)))
filt_plt.plot(freqs_filt, 10*np.log10(np.abs(filt_spect)))
filtrd_plt.plot(freqs, 10*np.log10(np.abs(filtrd_spect)))
orig_plt.set_ylim([0, orig_plt.get_ylim()[1]])
filtrd_plt.set_ylim([0, filtrd_plt.get_ylim()[1]])
orig_plt.set_xscale("log")
orig_plt.grid()
filt_plt.grid()
filtrd_plt.grid()
orig_plt.set_title("Frequency Responses")
orig_plt.set_ylabel("Original dB")
filt_plt.set_ylabel("Filter dB")
filtrd_plt.set_ylabel("Filtered dB")
filtrd_plt.set_xlabel("$f$ (Hz)")


# Volume match filtered audio
filtered = filtered * np.max(audio)/np.max(filtered)

# Write audio
print("Writing audio...")
filtered = filtered.astype(original_type)
wavfile.write("software_filtered_audio.wav", rate=f_samp, data=filtered)

# Wait on plots to finish
print("Finalizing plots...")
plt.pause(5)

# Play audio
print("Playing audio!")
sd.play(filtered, samplerate=f_samp)
input("Press ENTER to stop playing.")
sd.stop()
input("Press ENTER to exit.")