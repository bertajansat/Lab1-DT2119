import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from scipy.fft import fft, dct
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

from lab1_tools import trfbank, lifter, tidigit2labels

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
    _, windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    _, log_mel_spectrum = logMelSpectrum(spec, samplingrate)
    return log_mel_spectrum


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


def dtw(x, y, dist=distance_matrix):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of shape (N, D) and (M, D) respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences (number of frames)
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path through AD 

    Note that you only need to define the first output for this exercise.
    """

    # number of frames in each utterance
    N, M = x.shape[0], y.shape[0]

    # computation of local distance matrix LD
    LD = dist(x, y)

    # computation of accumulated distance matrix AD (DTW algorithm)
    AD = np.full_like(LD, np.inf)
    AD[0, :] = np.cumsum(LD[0, :])      # first row can only come from stretching y (from left)
    AD[:, 0] = np.cumsum(LD[:, 0])      # first column can only come from stretching x (from top)

    for i in range(N):
        for j in range(M):
            AD[i,j] = LD[i, j] + min(
                AD[i-1, j],
                AD[i, j-1],
                AD[i-1, j-1],
            )

    global_dist = 1 / (N + M) * AD[-1, -1]

    # backtracing (always picking lowest accumulated cost predecessor)
    optimal_path = [(N-1, M-1)]
    i, j = N-1, M-1

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (AD[i-1, j-1], i-1, j-1),  
                (AD[i-1, j],   i-1, j),     
                (AD[i, j-1],   i,   j-1),   
            ]
            _, i, j = min(candidates, key=lambda c: c[0])
        optimal_path.append((i, j))
    
    optimal_path.reverse()      # to have "start-to-finish" semantics

    return global_dist, LD, AD, optimal_path



# SECTION 4 ----------------------------------------

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
# fig, ax = plt.subplots(figsize=(10, 4))
# mesh = ax.pcolormesh(time_axis, np.arange(mfcc_features.shape[1]), mfcc_features.T, shading='auto')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Cepstral coefficient index')
# ax.set_title('MFCC coefficients (before liftering)')
# plt.colorbar(mesh, ax=ax, label='MFCC value')
# plt.tight_layout()
# plt.savefig('mfcc_test.png')
# print(f'DCT results are matching: {np.allclose(mfcc_features, example['mfcc'])}')

lmfcc_features = lifter(mfcc_features)  # liftering = sinusoidal weighting to equalise feature scales
# fig, ax = plt.subplots(figsize=(10, 4))
# mesh = ax.pcolormesh(time_axis, np.arange(lmfcc_features.shape[1]), lmfcc_features.T, shading='auto')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Cepstral coefficient index')
# ax.set_title('MFCC coefficients (after liftering)')
# plt.colorbar(mesh, ax=ax, label='Liftered MFCC value')
# plt.tight_layout()
# plt.savefig('lmfcc_test.png')
# print(f"\nLiftering effect (std per coefficient)")      # std brought to approx. same order of magnitude
# print(f"{'Coeff':<8} {'Before':>10} {'After':>10} {'Ratio':>8}")
# for i in [0, 4, 8, 12]:
#     before = mfcc_features[:, i].std()
#     after = lmfcc_features[:, i].std()
#     print(f"  {i:<6} {before:>10.2f} {after:>10.2f} {after/before:>8.2f}x")
# print(f'Liftering results are matching: {np.allclose(lmfcc_features, example['lmfcc'])}')



# SECTION 5 --------------

data = np.load('lab1_data.npz', allow_pickle=True)['data']
labels = tidigit2labels(data)

# main win if features are uncorrelated: diagonal covariance matrix (when using multivariate Gaussians)
# -> only M parameters for covariance instead of M*(M+1)/2 parameters (per mixture component)
# -> each feature dimension can be modeled independently via a Gaussian (mixture)
log_mel_all = []
lmfcc_all = []

for utterance in data:
    samples = utterance['samples'].astype(np.float64)
    log_mel_all.append(mspec(samples))
    lmfcc_all.append(mfcc(samples))


# concatenate log Mel scale and liftered MFCC results into (N,M) matrices respectively 
# -> N: total number of frames in 44 utterances dataset
# -> M: number of features
log_mel_concat = np.vstack(log_mel_all)     # (total_nr_frames x 40) = (3885 x 40)
lmfcc_concat = np.vstack(lmfcc_all)         # (total_nr_frames x 13) = (3885 x 40)


# computation of Pearson correlation matrices
corr_mspec = np.corrcoef(log_mel_concat.T)  # 40 x 40 matrix
corr_lmfcc = np.corrcoef(lmfcc_concat.T)    # 13 x 13 matrix 


# plotting as heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# log Mel features correlation
im0 = axes[0].pcolormesh(corr_mspec, cmap='coolwarm', vmin=-1, vmax=1, shading='auto')
axes[0].set_xlabel('Mel filterbank index')
axes[0].set_ylabel('Mel filterbank index')
axes[0].set_title('(Log) Mel filterbank correlation')
axes[0].set_aspect('equal')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# liftered MFCC correlation
im1 = axes[1].pcolormesh(corr_lmfcc, cmap='coolwarm', vmin=-1, vmax=1, shading='auto')
axes[1].set_xlabel('MFCC coefficient')
axes[1].set_ylabel('MFCC coefficient')
axes[1].set_title('MFCC correlation (13×13)')
axes[1].set_aspect('equal')
plt.colorbar(im1, ax=axes[1], shrink=0.8)

plt.tight_layout()
# plt.savefig('correlation_comparison.png')



# SECTION 6 --------------

# for decorrelated/independent features (MFCC), diagonal covariance matrix assumption is justified
# -> (liftered) MFCC makes diagonal covariance GMMs feasible compared to full-covariance GMMs
# -> trying to use GMM clustering (several multivariate Gaussian components, each representing a cluster) 
#    for identifying speech segments (sound categorisation, e.g. into phonemes)
# -> after training, we can use the learned GMM probability distribution to compute posteriors 
#    (probability that a certain frame corresponds to a certain component, ideally representative for a certain sound)
# -> posterior distribution over time for an utterance shows which component (that is, sound) is active at some moment

# training of diagonal GMMs with varying number of components
nr_components_list = [4, 8, 16, 32]
gmm_models = {}

for K in nr_components_list:
    gmm = GaussianMixture(n_components=K, covariance_type='diag', random_state=26)
    gmm.fit(lmfcc_concat)
    gmm_models[K] = gmm


# compute posteriors with different GMM models for one utterance ("seven")
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

utt_idx = 16
utt_lmfcc = lmfcc_all[utt_idx]

for ax, K in zip(axes, nr_components_list):
    posteriors = gmm_models[K].predict_proba(utt_lmfcc)         # (n_frames, K)
    ax.pcolormesh(posteriors.T, cmap='viridis', shading='auto')
    ax.set_ylabel(f'{K} components')
    ax.set_yticks([])

axes[0].set_title(f'GMM posteriors for "{labels[utt_idx]}" – varying number of components')
axes[-1].set_xlabel('Frame index')
plt.tight_layout()
# plt.savefig('single_utterance_varying_K.png')


# 32-component GMM posteriors for the four "seven" utterances
seven_indices = [16, 17, 38, 39]
gmm32 = gmm_models[32]

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

for ax, utt_idx in zip(axes, seven_indices):
    utt_lmfcc = lmfcc_all[utt_idx]
    posteriors = gmm32.predict_proba(utt_lmfcc)
    ax.pcolormesh(posteriors.T, cmap='viridis', shading='auto')
    ax.set_ylabel(labels[utt_idx], fontsize=10)
    ax.set_yticks([])

axes[0].set_title('GMM posteriors (32 components) — four "seven" utterances')
axes[-1].set_xlabel('Frame index')
plt.tight_layout()
# plt.savefig('single_K_several_utterances.png')



# SECTION 7 --------------

# comparison of utterances might cause problems even when saying the same word (alignment, length, ...)
# -> dynamic time warping finds optimal (minimum cost) alignment between two sequences
# -> uses stretching of either utterance or synchronous advancing of both

# computation of pairwise local Euclidean distances and global distances between utterances
nr_utterances = len(lmfcc_all)
LD = np.empty((nr_utterances, nr_utterances), dtype=object)
GD = np.empty((nr_utterances, nr_utterances))

for i in range(nr_utterances):
    for j in range(i+1, nr_utterances):
        gd, ld, _, _ = dtw(lmfcc_all[i], lmfcc_all[j])

        LD[i, j] = ld       # 44x44 matrix, each element storing pairwise Euclidean distance matrix 
        LD[j, i] = ld.T

        GD[i, j] = gd       # 44x44 matrix, each element storing a global distance scalar
        GD[j, i] = gd

fig, ax = plt.subplots(figsize=(10, 9))
mesh = ax.pcolormesh(GD, cmap='viridis', shading='auto')
ax.set_xticks(np.arange(nr_utterances) + 0.5)
ax.set_yticks(np.arange(nr_utterances) + 0.5)
ax.set_xticklabels(labels, rotation=90, fontsize=6)
ax.set_yticklabels(labels, fontsize=6)
ax.set_title('Pairwise DTW global distance matrix (44 utterances)')
ax.set_aspect('equal')
plt.colorbar(mesh, ax=ax, label='DTW distance', shrink=0.8)
plt.tight_layout()
plt.savefig('global_distances.png')


# performing hierarchical clustering
GD_condensed = squareform(GD)   # upper triangular matrix elements of pairwise distance matrix
Z = linkage(GD_condensed, method='complete')

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=7, ax=ax)
ax.set_title('Hierarchical clustering of utterances (complete linkage)')
ax.set_ylabel('DTW distance')
plt.tight_layout()
plt.savefig('dendrogram.png')