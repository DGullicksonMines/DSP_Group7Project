# DSP Group 7 Final Project Software Package

The purpose of this software package is creating a corrective filter from a room's frequency response using simple methods.

The larger repository this directory is contained within may be found at https://github.com/DGullicksonMines/DSP_Group7Project

This directory contains four (4) Python programs (`*.py`), two (2) text files (`*.txt`), and two (2) audio files (`*.wav`) of note.

`room_response.py` may be run at any time to produce a plot of the room response that's recorded in `AvgFreqResponse.txt`, which was produces by commercial software utilizing a calibrated microphone in the lecture room.

`corrective_filter.py` may be run before `play.py` to produce a corrective filter from the room response that's recorded in `AvgFreqResponse.txt` and the target response recorded in `Harman Target Curve.txt`, then will store the filter data in a file named `corr_filt.npy` for use by `play.py`.

`calibration.py` may be run before `play.py` as an alternative way of producing `corr_filt.npy` by simulaneously playing and recording `calibration.wav`. This method, however, does not match the target curve, instead matching to a completely flat response.

**NOTE**: You should ensure that Acoustic Echo Cancellation (AEC) is turned *OFF* on your device while `calibration.py` runs, otherwise the played audio may not be properly recorded. Moreover, this method will work more poorly, depending on the quality of the utilized audio input and output devices.

`play.py` may be run after `corr_filt.npy` has been produced to apply the corrective filter to `Audio.wav` before writing the corrected audio to `software_filtered_audio.wav` and playing it.