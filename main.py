import functools
import io
import subprocess
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from numba import njit, prange
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from features import (get_abs_amp_features, get_mb_entropy, get_mb_onset,
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

@njit(cache=True)
def filter_abs_difference(abs_difference, threshold=2):
    sums = abs_difference.sum(axis=0)
    for i in range(abs_difference.shape[0]):
        for k in range(abs_difference.shape[1]):
            if sums[k] < threshold:
                abs_difference[i, k] = 0
    return abs_difference


@njit(cache=True)
def roll_to_normalize_freq(amplitudes):
    length, bands = amplitudes.shape
    out = np.empty_like(amplitudes)
    for i in range(length):
        first = 0
        for k in range(bands):
            if amplitudes[i, k] >= 0.0002:
                first = k
                break
        for k in range(bands):
            out[i, k] = amplitudes[i, (k + first) % bands]
    return out


def multiline(d):
    plt.figure()
    plt.plot(d.T)
    plt.xlabel("Sample index")
    plt.ylabel("Absolute difference")
    plt.grid(True)
    plt.show()


def hdbscan_difference(abs_differences, min_cluster_size=10, min_samples=5, cluster_selection_epsilon=0.0, alpha=1.0, metric="euclidean", algorithm="auto", power=0.5):
    if min_cluster_size < 2 or min_samples < 1:
        raise ValueError("min_cluster_size must be at least 2 and min_samples at least 1")
    if len(abs_differences) < max(min_cluster_size, min_samples):
        return np.full(len(abs_differences), -1, dtype=int)
    x = abs_differences.reshape(len(abs_differences), -1)
    x = np.sign(x) * np.abs(x) ** power
    X_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    return HDBSCAN(
        algorithm=algorithm,
        metric=metric,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="leaf"
    ).fit_predict(X_norm)


def plot_clusters_2d(abs_differences, labels):
    abs_differences = abs_differences.reshape(len(abs_differences), -1)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(abs_differences)

    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=1, linewidths=0)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

@njit(cache=True, parallel=True)
def compute_similarity(x):
    n, bands = x.shape
    similarity = np.zeros((n, n))
    for i in prange(n):
        for j in range(i + 1, n):
            intersection = 0.0
            union = 0.0
            for k in range(bands):
                intersection += min(x[i, k], x[j, k])
                union += max(x[i, k], x[j, k])
            similarity[i, j] = (intersection + 1e-8) / (union + 1e-6)
    return similarity


def get_similarity(abs_difference, threshold=0.5):
    x = np.ascontiguousarray(abs_difference, dtype=np.float64) ** 3
    similarity = compute_similarity(x)
    return similarity, *np.where(np.triu(similarity > threshold, k=1))


@njit(cache=True)
def get_stems(amplitudes, frequencies, rows, cols):
    groups = 0
    for i in range(rows.size):
        if i == 0 or rows[i] != rows[i - 1]:
            groups += 1
    out = np.full((groups, amplitudes.shape[1]), np.inf)
    group = -1
    for i in range(rows.size):
        if i == 0 or rows[i] != rows[i - 1]:
            group += 1
        for k in range(amplitudes.shape[1]):
            out[group, k] = np.minimum(out[group, k], amplitudes[cols[i], k])
    return out


@njit(cache=False)
def filter_stems(stems, rows, cols, level=0.25, counts=10, min_distance=100):
    distant_counts = np.zeros(stems.shape[0], dtype=np.int64)
    group = -1
    for i in range(rows.size):
        if i == 0 or rows[i] != rows[i - 1]:
            group += 1
        if abs(cols[i] - rows[i]) > min_distance:
            distant_counts[group] += 1
    mask = (stems.sum(axis=1) > level) & (distant_counts > counts)
    return stems[mask], mask

@njit(cache=True, parallel=True)
def _stem_presence(amplitudes, stems):
    present = np.empty((stems.shape[0], amplitudes.shape[0]), dtype=np.bool_)
    for j in prange(stems.shape[0]):
        for i in range(amplitudes.shape[0]):
            present[j, i] = True
            for k in range(amplitudes.shape[1]):
                if not amplitudes[i, k] >= stems[j, k]:
                    present[j, i] = False
                    break
    return present


def find_stems(normalized_freq_pitches, stems, stems_power):
    amplitudes = np.power(normalized_freq_pitches, stems_power)
    present = _stem_presence(amplitudes, stems)
    return [np.flatnonzero(row) for row in present]


# def gather_stems(group_stems, similarity_matrix):
#     unique = np.unique(functools.reduce(lambda a, b: np.hstack((a, b)), group_stems))
#     similarity_matrix[unique[:, None], unique[None, :]]
#     cumulated_score = 0
#     for group in group_stems:
#         for stem in group:
#
#     return stems


def play_segments(indices, audio, sample_rate, hop_ms=10, duration=0.25, frames_per_group=2):
    """Build browser-playable audio from consecutive detection excerpts."""
    starts = [int(index * frames_per_group * hop_ms * sample_rate / 1000) for index in indices]
    clips = [audio[start:start + round(duration * sample_rate)] for start in starts]
    output = io.BytesIO()
    wavfile.write(output, sample_rate, np.concatenate(clips).astype(np.float32))
    return output.getvalue()
#
#
# def filter_clusters(labels, mb_onset, n_clusters):
#     argsort = labels.argsort()
#     sorted_labels = labels[argsort]
#     indices = np.hstack(([0] ,np.flatnonzero(sorted_labels[:-1] != sorted_labels[1:]) + 1, labels.shape[-1] - 1))
#     avg_onsets = []
#     for b, n in pairwise(indices):
#         avg_onsets.append(np.mean(mb_onset.T[argsort][b:n]))
#     keep = np.argsort(avg_onsets)[:n_clusters]
#     return np.where(np.isin(labels, keep), labels, -1)

def recognize_stems(
        audio,
        sample_rate,
        frames_per_group=2,
        progress=None,
        n_bands=200,
        min_cluster_size=10,
        min_samples=5,
        split_size=0.05,
        cluster_selection_epsilon=0.0,
        alpha=1.0,
        metric="euclidean",
        algorithm="auto",
        power=0.5,
):
    if n_bands < 1:
        raise ValueError("n_bands must be positive")
    if min_cluster_size < 2 or min_samples < 2:
        raise ValueError("min_cluster_size must be at least 2 and min_samples at least 2")
    step = 0

    def run(label, function, *args, **kwargs):
        nonlocal step
        if progress is not None:
            progress(step / 14, label)
        result = function(*args, **kwargs)
        step += 1
        return result

    frequencies, amplitudes, spectrum, fft_bands = run("Computing the Fourier transform…", fourier_transform, audio, sample_rate, n_bands=n_bands, return_spectrum=True)
    average_grouped_amplitudes = run("Grouping the amplitudes frames…", group_amplitudes_frames, amplitudes, frames_per_group)
    abs_differences = run("Getting frames absolute difference…", get_abs_difference, average_grouped_amplitudes)
    max_amplitude, count_amplitudes, avg_amplitude_fq, std_amplitudes = run("Computing amplitude features…", get_abs_amp_features, abs_differences, frequencies)
    mb_entropy = run("Computing band entropy…", get_mb_entropy, average_grouped_amplitudes, split_size=split_size)
    mb_onset = run("Computing band onset…", get_mb_onset, average_grouped_amplitudes, split_size=split_size)
    mb_s_centroid = run("Computing band centroids…", get_mb_s_centroid, average_grouped_amplitudes, frequencies, split_size=split_size)
    mb_s_flux = run("Computing spectral flux…", get_mb_s_flux, average_grouped_amplitudes, split_size=split_size)
    mb_zcr = run("Computing row sign changes…", get_mb_zcr, average_grouped_amplitudes, split_size=split_size)
    mb_mid_side_e = run("Computing stereo mid/side energy…", get_mb_mid_side_e, spectrum, fft_bands, n_bands, split_size=split_size)
    mb_mid_side_e = run("Grouping mid/side energy…", group_amplitudes_frames, mb_mid_side_e.T, frames_per_group).T
    mb_crest_factor = run("Computing crest factors…", get_mb_crest_factor, average_grouped_amplitudes, split_size=split_size)
    # Match the original frame alignment and the length of abs_differences.
    mb_entropy, mb_onset, mb_s_centroid, mb_s_flux, mb_zcr, mb_mid_side_e, mb_crest_factor = (
        feature[:, :-1] for feature in
        (mb_entropy, mb_onset, mb_s_centroid, mb_s_flux, mb_zcr, mb_mid_side_e, mb_crest_factor)
    )
    features = np.vstack((
        max_amplitude[None, :],
        count_amplitudes[None, :],
        avg_amplitude_fq[None, :],
        std_amplitudes[None, :],
        mb_entropy,
        mb_onset,
        mb_s_centroid,
        mb_s_flux,
        mb_zcr,
        mb_mid_side_e,
        mb_crest_factor,
    )).T

    labels = run(f"Clustering differences…", hdbscan_difference, features, min_cluster_size, min_samples, cluster_selection_epsilon, alpha, metric, algorithm, power)
    # f_labels = run(f"Filtering clusters into {n_clusters} groups…", filter_clusters, labels, mb_onset, n_clusters)

    run("Plotting clusters…", plot_clusters_2d, features, labels)

    if progress is not None:
        progress(1.0, "Analysis complete")
    return labels
