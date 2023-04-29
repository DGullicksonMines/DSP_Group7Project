#NOTE If the played audio is not being detected, ensure that
# Acoustic Echo Correction (AEC) is turned OFF in your sound settings.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fft as fft
import scipy.io.wavfile as wavfile
from scipy.constants import pi

# Documentation: https://python-sounddevice.readthedocs.io/en/0.4.6/usage.html
import sounddevice as sd

# Load calibration audio
# f_samp, calibration = wavfile.read("implementation/problem3.wav") #TODO use actual calibration audio
f_samp, calibration = wavfile.read("adaptive_filter_testing/Audio/Radioactive.wav") #TODO use actual calibration audio

# Remove all but one channel from the audio
if len(calibration.shape) > 1: calibration = calibration[:, 0]

# Fix audio
# NOTE This part isn't needed for the actual calibration audio
duration_ms = 2000
duration_samps = f_samp * duration_ms // 1000
calibration = calibration[:duration_samps]



# ---== Get room response ==---
response = sd.playrec(calibration, samplerate=f_samp, channels=2, blocking=True)
response = response[:, 1]
resp_len = len(response)
# response = np.reshape(response, newshape=(resp_len,))

# Volume match response
response = response.astype(np.float64) * np.max(calibration)/np.max(response)

# Create lowpass filter for response
# TODO detect cutoff frequency automatically
lp_b, lp_a = sig.iirdesign(wp=15500, ws=16000, gpass=0.1, gstop=50, fs=f_samp)
lp_freqs, lp_spect = sig.freqz(b=lp_b, a=lp_a, worN=1024, fs=f_samp)
_, (impulse,) = sig.dimpulse(system=(lp_b, lp_a, 1/f_samp), n=200)

response = sig.filtfilt(b=lp_b, a=lp_a, x=response)

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



# Plot signals
duration = resp_len/f_samp
t = np.linspace(0, duration, resp_len)
plt.ion()
_, (p1, p2) = plt.subplots(2, sharex=True, layout="constrained")
p1.plot(t, calibration)
p2.plot(t, response)
p1.set_title("Signals")
p1.set_ylabel("Actual")
p2.set_ylabel("Response")
p2.set_xlabel("$t$ (seconds)")

# Plot spectrums
freqs = np.linspace(-f_samp/2, f_samp/2, resp_len)
actual_spect = fft.fftshift(fft.fft(calibration))
resp_spect = fft.fftshift(fft.fft(response))
actual_abs = np.abs(actual_spect)
resp_abs = np.abs(resp_spect)

_, (actual_mag, resp_mag) = plt.subplots(2, sharex=True, layout="constrained")
actual_mag.plot(freqs, 10*np.log10(actual_abs))
resp_mag.plot(freqs, 10*np.log10(resp_abs))
actual_mag.set_title("Frequency Spectums")
actual_mag.set_ylabel("Actual dB")
resp_mag.set_ylabel("Response dB")
resp_mag.set_xlabel("$f$ (Hz)")



# Run matched filter on response
filt = calibration[::-1]

filtered = sig.convolve(response, filt)
sig_len = len(filtered)
fltrd_duration = sig_len/f_samp
t = np.linspace(0, fltrd_duration, sig_len)

# Plot the output of the matched filter
plt.figure()
plt.title("Matched Filter Output")
plt.xlabel("Time (s)")
plt.plot(t, filtered)



# Perform calibration
#NOTE Some things that may help could be:
# - Reducing the range of frequencies the filter attenuates.
attenuation = actual_spect/resp_spect
#TODO calculate this automatically
attenuation[np.abs(freqs) > 10000] = 1

# Manually smooth attenuation
ranges = 20
oddity = int(len(attenuation) % 2)
attenuation = fft.ifftshift(attenuation)
attenuation = attenuation[:(len(attenuation)+oddity)//2]
range_len = len(attenuation)//ranges
for i in range(ranges):
    start = range_len*i
    end = range_len*(i+1)
    attenuation[start:end] = np.mean(attenuation[start:end])
if oddity == 0:
    attenuation = np.concatenate((attenuation[::-1], attenuation))
else:
    attenuation = np.concatenate((attenuation[:0:-1], attenuation))

calib_filt = fft.ifft(fft.ifftshift(attenuation))
calib_filt = np.real(calib_filt)
calib_filt = calib_filt[:255] # Reduce number of taps

# Make bode plots
mag = np.abs(attenuation)
phase = np.angle(attenuation)
fig, (impulse_plot, mag_plot, phase_plot) = plt.subplots(3, layout="constrained")
impulse_plot.stem(calib_filt)
mag_plot.plot(freqs, 10*np.log10(mag))
phase_plot.plot(freqs, phase)

impulse_plot.set_title("Filter Impulse Response")
impulse_plot.set_xlabel(r"$n$")
mag_plot.set_title("Filter Frequency Response")
mag_plot.set_ylabel(r"$|H(\omega)|$ (dB)")
mag_plot.sharex(phase_plot)
mag_plot.xaxis.set_visible(False)
phase_plot.set_ylabel(r"$\angle H(\omega)$ (rad)")
phase_plot.set_xlabel(r"$f$ (Hz)")



plt.show()
if __name__ == "__main__":
    input("Press ENTER to exit.")