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
filtered = sig.filtfilt(b=c.calib_filt, a=1, x=audio)



# Create lowpass filter for response
# TODO detect cutoff frequency automatically
lp_b, lp_a = sig.iirdesign(wp=6000, ws=6500, gpass=0.1, gstop=50, fs=f_samp)
lp_freqs, lp_spect = sig.freqz(b=lp_b, a=lp_a, worN=1024, fs=f_samp)
_, (impulse,) = sig.dimpulse(system=(lp_b, lp_a, 1/f_samp), n=200)

filtered = sig.filtfilt(b=lp_b, a=lp_a, x=filtered)

# Make bode plots
mag = np.abs(lp_spect)
phase = np.angle(lp_spect)
fig, (impulse_plot, mag_plot, phase_plot) = plt.subplots(3, layout="constrained")
impulse_plot.stem(impulse)
mag_plot.plot(lp_freqs, 10*np.log10(mag))
phase_plot.plot(lp_freqs, phase)

impulse_plot.set_title("Lowpass Impulse Response")
impulse_plot.set_xlabel(r"$n$")
mag_plot.set_title("Lowpass Frequency Response")
mag_plot.set_ylabel(r"$|H(\omega)|$ (dB)")
mag_plot.sharex(phase_plot)
mag_plot.xaxis.set_visible(False)
phase_plot.set_ylabel(r"$\angle H(\omega)$ (rad)")
phase_plot.set_xlabel(r"$f$ (Hz)")



# Plot signals and spectrums
audio_len = len(filtered)
duration = audio_len/f_samp
t = np.linspace(0, duration, audio_len)
_, (p1, p2) = plt.subplots(2, sharex=True, layout="constrained")
p1.plot(t, audio)
p2.plot(t, filtered)
p1.set_title("Signals")
p1.set_ylabel("Original")
p2.set_ylabel("Filtered")
p2.set_xlabel("$f$ (Hz)")

freqs = np.linspace(-f_samp/2, f_samp/2, audio_len)
orig_spect = fft.fftshift(fft.fft(audio))
filt_spect = fft.fftshift(fft.fft(filtered))
_, (p1, p2) = plt.subplots(2, sharex=True, layout="constrained")
p1.plot(freqs, np.abs(orig_spect))
p2.plot(freqs, np.abs(filt_spect))
p1.set_title("Spectrums")
p1.set_ylabel("Original")
p2.set_ylabel("Filtered")
p2.set_xlabel("$f$ (Hz)")
plt.show()



# Play audio
print("Playing audio!")
sd.play(filtered.astype(np.int16), samplerate=f_samp)
input("Press ENTER to exit.")