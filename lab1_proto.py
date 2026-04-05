import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from scipy.fft import fft, dct
import matplotlib.pyplot as plt

from lab1_tools import trfbank, lifter

### DT2119, LAB 1 FEATURE EXTRACTION ###


# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)


def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)


# Functions to be implemented ----------------------------------


def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows (using short-time stationarity assumption).

    Args:
        samples: sampled air pressure oscillations (sound amplitude over time)
        winlen: window length (in nr_samples = window_duration * sampling_rate)
        winshift: shift between consecutive windows (in number of samples)
    Returns:
        (N, winlen) numpy array, where N is the number of windows that fit
        in the input signal
    """

    # compute number of COMPLETE frames (discarding truncated one at the end)
    # -> no padding, no partial final frame
    N = (len(samples) - winlen) // winshift + 1

    # compute frame offsets for sliding-window approach
    win_idxs = np.arange(winlen)                     # indices within a single frame
    frames_offsets = np.arange(N) * winshift         # indices indicating the start of a frame
    frames_idxs = frames_offsets[:, None] + win_idxs[None, :]

    return samples[frames_idxs]

    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter. Pre-emphasis boosts higher frequencies since the glottal/voice source
    (vibration of vocal folds) produces a signal that is richer at lower frequencies. To avoid 
    leaving the power spectrum skewed towards lower frequencies and capturing important speech 
    information also at higher frequencies, pre-emphasis is used.

    Args:
        input: (N, winlen) array of speech frames, where N is the number of frames and
               winlen the number of samples per frame
        p: preemphasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

    # pre-emphasis is first-order (linear), high-pass (allow high freq.) filter
    # -> emphasize sudden changes (high freq.) and attenuate slow changes (low freq.)
    # -> relative flattening of spectrum 
    # -> lfilter coefficients (b, a) chosen to obtain pre-emphasis formula y[n] = x[n] - a*x[n-1]
    # -> lfilter assumes zero initial conditions for y[0] = x[0]
    return lfilter(b=(1, -p), a=(1), x=input, axis=1)


def windowing(input):
    """
    Applies Hamming window to the input frames.

    Args:
        input: (N, winlen) array of speech samples, where N is the number of frames and
               winlen the samples per frame
    Output:
        (N, winlen) array of windowed speech samples
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

    # enframing of original signal corresponds to multiplication with rectangular window
    # -> likely introduces sharp discontinuity at frame boundaries 
    # -> spectral leakage since Fourier transform assumes periodicity of frame signal
    # -> frequency spectrum gets "smeared" with spurious energy across all frequencies

    # Hamming windowing preserves samples around frame center, those near both edges are faded out
    winlen = input.shape[1]
    hamming_win = hamming(winlen, sym= False)

    return hamming_win, input * hamming_win


def powerSpectrum(input, nfft=512):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: (N, winlen) array of speech samples, where N is the number of frames and
               winlen the samples per frame
        nfft: length of the FFT
    Output:
        (N, nFFT) array of power spectra
    Note: you can use the function fft from scipy.fftpack
    """

    # speech sounds fundamentally distinguished by frequency domain content (pitch+timbre)
    # -> change from time domain to frequency domain needed
    # -> decompose time signal into sum of periodic functions at different frequencies (Fourier transform)
    # -> Fourier transform components contain information about how much energy is contained at each frequency
    # -> phase/timing of each frequency component irrelevant for energy (vanishes with squared magnitude)

    # use Fast Fourier Transform (FFT) algo for computing DFT of each signal frame
    # -> by Nyquist-Shannon sampling theorem, maximum representable freq is half of sampling rate (i.e. 10k Hz)
    # -> first 257 FFT bins (including 0 and f_max) are unique, rest is mirrored result (bin resolution of 20k/512)
    fft_components = fft(input, n=nfft, axis=1)     # since nfft > winlen=400, padding with zeros 

    return np.abs(fft_components)**2


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: (N, nfft) array of power spectrum coefficients, where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        (N, n_melfilters) array of Mel filterbank log outputs, where n_nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """

    # previous (linearly spaced) DFT bins treat all frequency equally, unlike human auditory perception!!!
    # -> Mel filterbank reshapes spectrum to match human perception (logarithmic perception rather than linear)
    # -> Mel scale roughly linear below 1k Hz and logarithmic above
    # -> since bank of triangular filters evenly spaced on Mel scale: 
    #    densely packed narrow filters at low-freq, sparse wide filters at high-freq values on frequency scale
    # -> each filter acts like weighted average over a certain range of FFT bins (linear frequency scale)
    mel_filterbank = trfbank(samplingrate, input.shape[1])
    mel_features = input @ mel_filterbank.T

    return mel_filterbank, np.log(mel_features)   # taking log here is due to logarithmic human loudness perception


def cepstrum(input, nceps):
    """
    Calulates cepstral coefficients from log Mel spectrum applying Discrete Cosine Transform

    Args:
        input: (N, n_melfilters) array of log outputs of Mel scale filterbank, where N is the
               number of frames and n_melfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        (N, n_ceps) array of cepstral coefficients
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    # adjacent Mel spectrum features within a frame are highly correlated (neighbouring filters overlap)
    # -> Discrete Cosine Transform (DCT) decorrelates these features and makes them independent:
    #    decomposes log Mel spectrum into set of cosine base functions
    # -> low-order DCT features capture smooth, broad spectrum shape (vowels vs consonants), high-oder ones 
    #    the fine rapid fluctuations (noise, speaker-specific pitch harmonics)
    # -> correlated features would require more expensive modelling (more parameters, heavier training, more data, ...)
    # -> keeping nceps of Mel features (13 < 40) after DCT results in compression + spectral detail loss in word recognition
    return dct(input, type=2, axis=1)[:, :nceps]    # no normalisation, default DCT type, only slicing afterwards !


def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """


# Testing script ----------------------------------------

example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

win_dur = 0.02                              # window/frame duration (in s)
win_timeshift = 0.01                        # temporal shift between consecutive windows
sampling_rate = example['samplingrate']     # sampling rate (in s)

win_len = int(win_dur * sampling_rate)           # number of samples per window
win_shift = int(win_timeshift * sampling_rate)   # number of samples between consecutive windows


### enframe check
frames = enframe(example['samples'], win_len, win_shift)
# fig, ax = plt.subplots()
# ax.pcolormesh(frames.T)
# plt.savefig('enframe_test.png')
# print(f'Enframe results are matching: {np.array_equal(frames, example['frames'])}')


### pre-emphasis check
preemph_frames = preemp(frames)
# print(f'Pre-emphasis results are matching: {np.array_equal(preemph_frames, example['preemph'])}')


### Hamming windowing check
hamming_win, hamming_frames = windowing(preemph_frames)
# fig, ax = plt.subplots()
# ax.plot(list(range(len(hamming_win))), hamming_win)
# plt.savefig('hamming_test.png')
# print(f'Hamming windowing results are matching: {np.allclose(hamming_frames, example['windowed'])}')


### FFT check
power_spectrum = powerSpectrum(hamming_frames)
# fig, ax = plt.subplots()
# ax.pcolormesh(power_spectrum.T)
# plt.savefig('fft_test.png')
# print(f'FFT power spectrum results are matching: {np.allclose(power_spectrum, example['spec'])}')


### Mel scale check
filterbank, log_mel_spectrum = logMelSpectrum(power_spectrum, sampling_rate)

freqs_axis = sampling_rate / 512 * np.arange(512)
# fig, ax = plt.subplots(figsize=(10, 4))
# for i in range(filterbank.shape[0]):
#     ax.plot(freqs_axis[:257], filterbank[i, :257], linewidth=0.9)
# ax.set_xlabel('Frequency (Hz)')
# ax.set_ylabel('Filter amplitude')
# ax.set_title('Mel filterbank (40 triangular filters on linear frequency scale)')
# ax.set_xlim(0, 8000)
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('filterbank_test.png')

time_axis = np.arange(log_mel_spectrum.shape[0]) * 200 / sampling_rate
# fig, ax = plt.subplots(figsize=(10, 4))
# mesh = ax.pcolormesh(time_axis, np.arange(log_mel_spectrum.shape[1]), log_mel_spectrum.T, shading='auto')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Mel filter index')
# ax.set_title('Log Mel filterbank spectrum')
# plt.colorbar(mesh, ax=ax, label='Log energy')
# plt.tight_layout()
# plt.savefig('logmelscale_test.png')

# print(f'Log Mel scale results are matching: {np.allclose(log_mel_spectrum, example['mspec'])}')


### Cepstrum + liftering check
mfcc_features = cepstrum(log_mel_spectrum, 13)
fig, ax = plt.subplots(figsize=(10, 4))
mesh = ax.pcolormesh(time_axis, np.arange(mfcc_features.shape[1]), mfcc_features.T, shading='auto')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Cepstral coefficient index')
ax.set_title('MFCC coefficients (before liftering)')
plt.colorbar(mesh, ax=ax, label='MFCC value')
plt.tight_layout()
plt.savefig('mfcc_test.png')
# print(f'DCT results are matching: {np.allclose(mfcc_features, example['mfcc'])}')

lmfcc_features = lifter(mfcc_features)  # liftering = sinusoidal weighting to equalise feature scales
fig, ax = plt.subplots(figsize=(10, 4))
mesh = ax.pcolormesh(time_axis, np.arange(lmfcc_features.shape[1]), lmfcc_features.T, shading='auto')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Cepstral coefficient index')
ax.set_title('MFCC coefficients (after liftering)')
plt.colorbar(mesh, ax=ax, label='Liftered MFCC value')
plt.tight_layout()
plt.savefig('lmfcc_test.png')
# print(f"\nLiftering effect (std per coefficient)")      # std brought to approx. same order of magnitude
# print(f"{'Coeff':<8} {'Before':>10} {'After':>10} {'Ratio':>8}")
# for i in [0, 4, 8, 12]:
#     before = mfcc_features[:, i].std()
#     after = lmfcc_features[:, i].std()
#     print(f"  {i:<6} {before:>10.2f} {after:>10.2f} {after/before:>8.2f}x")
# print(f'Liftering results are matching: {np.allclose(lmfcc_features, example['lmfcc'])}')