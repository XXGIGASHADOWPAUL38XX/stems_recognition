import functools
import io
import subprocess
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from numba import njit, prange
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from features import (normalize, get_multibands, get_abs_amp_features, get_mb_entropy, get_mb_onset,
                      get_mb_s_centroid, get_mb_s_flux, get_mb_zcr,
                      get_mb_mid_side_e, get_mb_crest_factor)

def read_audio(path):
    """Return sample rate and normalized stereo audio, shaped (samples, 2)."""
    wav = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "2", "-f", "wav", "-"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    sample_rate, audio = wavfile.read(io.BytesIO(wav))

    scale = np.iinfo(audio.dtype).max if np.issubdtype(audio.dtype, np.integer) else 1
    audio = audio.astype(np.float32) / scale
    return sample_rate, audio


@njit(cache=True, parallel=True)
def _sum_bands(fft_amplitudes, bands, n_bands):
    amplitudes = np.zeros((fft_amplitudes.shape[0], n_bands, 2))
    for i in prange(fft_amplitudes.shape[0]):
        for k in range(fft_amplitudes.shape[1]):
            for channel in range(2):
                amplitudes[i, bands[k], channel] += fft_amplitudes[i, k, channel]
    return amplitudes


def fourier_transform(audio, sample_rate, frame_ms=25, hop_ms=10, n_bands=200, return_spectrum=False):
    """Return stereo band magnitudes (frames, bands, 2), optionally raw complex FFT."""
    audio = np.asarray(audio)
    if audio.ndim == 1:
        audio = np.repeat(audio[:, None], 2, axis=1)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("audio must have shape (samples, 2)")
    n = round(sample_rate * frame_ms / 1000)
    hop = round(sample_rate * hop_ms / 1000)
    count = max(0, 1 + (len(audio) - n) // hop)
    frames = np.stack(
        [audio[i * hop:i * hop + n] for i in range(count)]
    ) if count else np.empty((0, n, 2), dtype=np.float32)
    frames *= np.hanning(n)[None, :, None]

    fft_frequencies = np.fft.rfftfreq(n, 1 / sample_rate)
    spectrum = 2 * np.fft.rfft(frames, axis=1) / n
    spectrum[:, 0] = 0  # ignore DC; retain stereo phase for mid/side energy
    fft_amplitudes = np.abs(spectrum)

    frequencies = np.geomspace(20, sample_rate / 2, n_bands)
    edges = np.r_[0, np.sqrt(frequencies[:-1] * frequencies[1:]), np.inf]
    bands = np.digitize(fft_frequencies, edges) - 1
    amplitudes = _sum_bands(fft_amplitudes, bands, n_bands)
    if return_spectrum:
        return frequencies, amplitudes, spectrum, bands
    return frequencies, amplitudes


def group_amplitudes_frames(amplitudes, nb_frames):
    if nb_frames <= 0:
        raise ValueError("nb_frames must be positive")
    margin = (-len(amplitudes)) % nb_frames
    if margin:
        amplitudes = np.concatenate((amplitudes, np.zeros((margin, *amplitudes.shape[1:]))))
    shape = (len(amplitudes) // nb_frames, nb_frames, *amplitudes.shape[1:])
    return amplitudes.reshape(shape).mean(axis=1)


@njit(cache=True)
def get_abs_difference(amplitudes):
    return np.clip(amplitudes[1:] - amplitudes[:-1], 0, 1)


def get_data_difference(data, split_size=0.2):
    difference = np.concatenate([
        band - band.mean(axis=(0, 1), keepdims=True) if band.size else band.copy()
        for band in get_multibands(data, split_size)
    ], axis=1)
    useless_values_indexes = np.where(np.all(difference == 0, axis=(0, 2)))[0]
    dims = np.where((np.isin(np.arange(data.shape[1]), useless_values_indexes)==False), True, False)
    return difference.transpose(1, 0, 2)[dims].transpose(1, 0, 2)

@njit(cache=True)
def filter_difference(difference, threshold=2):
    sums = difference.sum(axis=0)
    for i in range(difference.shape[0]):
        for k in range(difference.shape[1]):
            if sums[k] < threshold:
                difference[i, k] = 0
    return difference


def multiline(d):
    plt.figure()
    plt.plot(d.T)
    plt.xlabel("Sample index")
    plt.ylabel("Absolute difference")
    plt.grid(True)
    plt.show()



@njit(cache=True, parallel=True)
def _canopy_labels(x, order, t1, t2):
    n, dimensions = x.shape
    remaining = np.ones(n, dtype=np.bool_)
    centers = []
    for seed in order:
        if not remaining[seed]:
            continue
        center = np.zeros(dimensions)
        count = 0
        for i in range(n):
            squared_distance = 0.0
            for k in range(dimensions):
                squared_distance += (x[i, k] - x[seed, k]) ** 2
            distance = np.sqrt(squared_distance)
            if distance < t1:
                for k in range(dimensions):
                    center[k] += x[i, k]
                count += 1
            if distance < t2:
                remaining[i] = False
        centers.append(center / count)

    labels = np.empty(n, dtype=np.int64)
    for i in prange(n):
        closest, best = 0, np.inf
        for j in range(len(centers)):
            distance = 0.0
            for k in range(dimensions):
                distance += (x[i, k] - centers[j][k]) ** 2
            if distance < best:
                closest, best = j, distance
        labels[i] = closest
    return labels


def canopy_difference(differences, t1=1.0, t2=0.5):
    """Build overlapping Euclidean canopies, then assign to their nearest centroid."""
    if not np.isfinite(t1) or not np.isfinite(t2) or not 0 < t2 < t1:
        raise ValueError("Canopy thresholds must satisfy 0 < t2 < t1 and be finite")
    if len(differences) == 0:
        return np.empty(0, dtype=int)
    x = np.asarray(differences, dtype=float).reshape(len(differences), -1)
    if x.shape[1] == 0:
        return np.full(len(x), -1, dtype=int)
    if not np.isfinite(x).all():
        raise ValueError("Clustering input must be finite")
    order = np.random.default_rng(42).permutation(len(x))
    labels = _canopy_labels(np.ascontiguousarray(x), order, t1, t2)
    return np.unique(labels, return_inverse=True)[1]


def kmeans_difference(features, n_clusters):
    if n_clusters == 0:
        return np.full(len(features), -1, dtype=int)
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(features)


def pca_difference(differences):
    if len(differences) == 0:
        return np.empty((0, 3))
    differences = differences.reshape(len(differences), -1)
    if differences.shape[1] == 0:
        return np.empty((len(differences), 0))
    if len(differences) == 1:
        return np.zeros((1, 3))
    differences = np.ascontiguousarray(differences, dtype=np.float32)
    components = min(3, *differences.shape)
    embedding = PCA(n_components=components).fit_transform(differences)
    return np.pad(embedding, ((0, 0), (0, 3 - components)))


def plot_clusters_3d(X_3d, labels=None):
    if X_3d.size == 0:
        return
    ax = plt.figure().add_subplot(projection="3d")
    colors = {} if labels is None else dict(c=labels, cmap="tab10")
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=1, linewidths=0, **colors)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    plt.show()



def play_segments(indices, audio, sample_rate, hop_ms=10, duration=0.25, frames_per_group=2):
    """Build browser-playable audio from consecutive detection excerpts."""
    starts = [int(index * frames_per_group * hop_ms * sample_rate / 1000) for index in indices]
    clips = [audio[start:start + round(duration * sample_rate)] for start in starts]
    output = io.BytesIO()
    wavfile.write(output, sample_rate, np.concatenate(clips).astype(np.float32))
    return output.getvalue()
def recognize_stems(
        audio,
        sample_rate,
        frames_per_group=2,
        progress=None,
        n_bands=200,
        t1=1.0,
        t2=0.5,
        power=0.5,
        split_size=0.2,
):
    if n_bands < 1:
        raise ValueError("n_bands must be positive")
    step = 0

    def run(label, function, *args, **kwargs):
        nonlocal step
        if progress is not None:
            progress(step / 19, label)
        result = function(*args, **kwargs)
        step += 1
        return result

    frequencies, amplitudes, spectrum, fft_bands = run("Computing the Fourier transform…", fourier_transform, audio, sample_rate, n_bands=n_bands, return_spectrum=True)
    grouped = run("Grouping the amplitudes frames…", group_amplitudes_frames, amplitudes, frames_per_group)
    if len(grouped) < 2:
        raise ValueError("Audio must contain at least two grouped frames")
    abs_amplitudes = run("Computing absolute differences…", get_abs_difference, grouped)
    amplitude_differences = run("Centering amplitudes…", get_data_difference, grouped, split_size)
    amplitude_differences = run("Normalizing amplitude differences…", normalize, amplitude_differences, 1, power)
    differences = run("Centering absolute differences…", get_data_difference, abs_amplitudes, split_size)
    differences = run("Normalizing differences…", normalize, differences, 1, power)
    amplitude_features = run("Computing amplitude features…", get_abs_amp_features, grouped, frequencies, power=power)
    kwargs = dict(split_size=split_size, power=power)
    mb_entropy = run("Computing band entropy…", get_mb_entropy, grouped, **kwargs)
    mb_onset = run("Computing band onset…", get_mb_onset, grouped, **kwargs)
    mb_centroid = run("Computing band centroids…", get_mb_s_centroid, grouped, frequencies, **kwargs)
    mb_flux = run("Computing spectral flux…", get_mb_s_flux, grouped, **kwargs)
    mb_zcr = run("Computing row sign changes…", get_mb_zcr, grouped, **kwargs)
    mb_energy = run("Computing stereo mid/side energy…", get_mb_mid_side_e, spectrum, fft_bands, n_bands,
                    frames_per_group=frames_per_group, **kwargs)
    mb_crest = run("Computing crest factors…", get_mb_crest_factor, grouped, **kwargs)
    features = np.vstack((*amplitude_features, mb_entropy, mb_onset, mb_centroid,
                          mb_flux, mb_zcr, mb_energy, mb_crest)).T[1:]
    features = np.hstack((features, (amplitude_differences[1:].reshape(len(differences), -1))))
    # features = np.hstack((features, differences.reshape(len(differences), -1)))

    canopy_labels = run("Estimating cluster count with Canopy…", canopy_difference, features, t1, t2)
    n_clusters = len(np.unique(canopy_labels[canopy_labels >= 0]))
    labels = run(f"Clustering into {n_clusters} groups with KMeans…", kmeans_difference, features, n_clusters)
    coordinates = run("Computing PCA…", pca_difference, features)
    run("Plotting clusters…", plot_clusters_3d, coordinates, labels)

    if progress is not None:
        progress(1.0, "Analysis complete")
    return np.r_[-1, labels]  # The first frame has no preceding frame to compare.
