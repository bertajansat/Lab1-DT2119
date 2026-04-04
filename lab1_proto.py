import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from scipy.fft import fft
import matplotlib.pyplot as plt

### DT2119, Lab 1 Feature Extraction ###


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
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

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