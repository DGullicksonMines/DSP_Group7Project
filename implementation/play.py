import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fft as fft
import scipy.io.wavfile as wavfile
from scipy.constants import pi

# Documentation: https://python-sounddevice.readthedocs.io/en/0.4.6/usage.html
import sounddevice as sd

# Get audio
f_samp, audio = wavfile.read("audio/Radioactive.wav")
audio = audio[:, 0]

# Apply filter
calib_filt = np.load("implementation/calib_filt.npy")
print("Applying filter to audio...")
filtered = sig.filtfilt(b=calib_filt, a=1, x=audio)



# Create lowpass filter for response
# TODO detect cutoff frequency automatically
lp_b, lp_a = sig.iirdesign(wp=15500, ws=16000, gpass=0.1, gstop=50, fs=f_samp)
lp_freqs, lp_spect = sig.freqz(b=lp_b, a=lp_a, worN=1024, fs=f_samp)
_, (impulse,) = sig.dimpulse(system=(lp_b, lp_a, 1/f_samp), n=200)

filtered = sig.filtfilt(b=lp_b, a=lp_a, x=filtered)

# Volume match filtered audio
filtered = filtered * np.max(audio)/np.max(filtered)

# # Make bode plots
# mag = np.abs(lp_spect)
# phase = np.angle(lp_spect)
# fig, (impulse_plot, mag_plot, phase_plot) = plt.subplots(3, layout="constrained")
# impulse_plot.stem(impulse)
# mag_plot.plot(lp_freqs, 10*np.log10(mag))
# phase_plot.plot(lp_freqs, phase)

# impulse_plot.set_title("Lowpass Impulse Response")
# impulse_plot.set_xlabel(r"$n$")
# mag_plot.set_title("Lowpass Frequency Response")
# mag_plot.set_ylabel(r"$|H(\omega)|$ (dB)")
# mag_plot.sharex(phase_plot)
# mag_plot.xaxis.set_visible(False)
# phase_plot.set_ylabel(r"$\angle H(\omega)$ (rad)")
# phase_plot.set_xlabel(r"$f$ (Hz)")



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
p1.plot(freqs, 10*np.log10(np.abs(orig_spect)))
p2.plot(freqs, 10*np.log10(np.abs(filt_spect)))
p1.set_xlim([-20000, 20000])
p1.set_ylim([0, p1.get_ylim()[1]])
p2.set_ylim([0, p2.get_ylim()[1]])
p1.set_title("Audio Spectrums")
p1.set_ylabel("Original dB")
p2.set_ylabel("Filtered dB")
p2.set_xlabel("$f$ (Hz)")
plt.show()




# Write audio
print("Writing audio...")
filtered = filtered.astype(np.int16)
wavfile.write("audio/filtered_audio.wav", rate=f_samp, data=filtered)

# Wait on plots to finish
print("Finalizing plots...")
plt.pause(5)

# Play audio
print("Playing audio!")
sd.play(filtered, samplerate=f_samp)
input("Press ENTER to stop playing.")
sd.stop()
input("Press ENTER to exit.")