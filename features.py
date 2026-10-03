import numpy as np
from scipy.signal import lfilter

def to_mono(data):
    """Average the last channel axis of (frames, bins, channels) feature data."""
    data = np.asarray(data)
    return data.mean(axis=-1) if data.ndim == 3 else data


def get_splits(split_size=0.05):
    if not 0 < split_size <= 1:
        raise ValueError("split_size must be between 0 and 1")
    return np.arange(split_size, 1, split_size)

def get_multibands(amplitudes, split_size=0.05):
    splits = (amplitudes.shape[1] * get_splits(split_size)).astype(int)
    return np.split(amplitudes, splits, axis=1)

def normalize(data, axis, power=1.0):
    data = np.asarray(data, dtype=float)
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    if data.shape[axis] == 0:
        return np.zeros_like(data, dtype=float)

    def norm(values):
        low = np.min(values, axis=axis, keepdims=True)
        span = np.max(values, axis=axis, keepdims=True) - low
        return np.clip(np.divide(
            2 * (values - low) - span,
            span,
            out=np.zeros_like(values),
            where=span != 0,
        ), -1, 1)

    data = norm(data)
    return norm(np.sign(data) * np.abs(data) ** power)


def get_abs_amp_features(abs_amplitudes, frequencies, power=1.0):
    abs_amplitudes = to_mono(abs_amplitudes)
    total = abs_amplitudes.sum(axis=1)  # (frames,)
    avg_frequency = np.divide(
        (abs_amplitudes * frequencies[None, :]).sum(axis=1),
        total,
        out=np.zeros_like(total, dtype=float),
        where=total != 0,
    )

    return (
        normalize(np.max(abs_amplitudes, axis=1), axis=0, power=power),
        normalize(np.count_nonzero(abs_amplitudes, axis=1), axis=0, power=power),
        normalize(avg_frequency, axis=0, power=power),
        normalize(np.std(abs_amplitudes, axis=1), axis=0, power=power),
    )


def get_mb_entropy(amplitudes, split_size=0.05, power=1.0):
    amplitudes = to_mono(amplitudes)
    result = []
    for band in get_multibands(amplitudes, split_size):
        total = band.sum(axis=1, keepdims=True)
        p = np.divide(band, total, out=np.zeros_like(band, dtype=float), where=total != 0)
        logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
        result.append(-(p * logp).sum(axis=1))
    return normalize(np.array(result), axis=1, power=power)


def get_mb_onset(amplitudes, split_size=0.05, power=1.0):
    """Positive flux minus noise floor; output is (multibands, frames)."""
    flux = _mb_s_flux(amplitudes, split_size)
    if not flux.shape[1]:
        return flux
    floor = lfilter([0.05], [1, -0.95], flux, axis=1)
    return normalize(np.maximum(flux - floor, 0), axis=1, power=power)


def get_mb_s_centroid(amplitudes, frequencies=None, split_size=0.05, power=1.0):
    """Centroid computed from frequencies (or bin indices), then normalized over time."""
    amplitudes = to_mono(amplitudes)
    mb = get_multibands(amplitudes, split_size)
    frequencies = np.arange(amplitudes.shape[1]) if frequencies is None else np.asarray(frequencies)
    freq_shape = (1, len(frequencies)) + (1,) * (amplitudes.ndim - 2)
    frequency_bands = get_multibands(frequencies.reshape(freq_shape), split_size)
    return normalize(np.array([
        np.divide((band * freq).sum(axis=1), band.sum(axis=1),
                  out=np.zeros_like(band.sum(axis=1), dtype=float), where=band.sum(axis=1) != 0)
        for band, freq in zip(mb, frequency_bands)
    ]), axis=1, power=power)


def get_mb_s_flux(amplitudes, split_size=0.05, power=1.0):
    """Positive frame-to-frame spectral change, normalized per band over time."""
    return normalize(_mb_s_flux(amplitudes, split_size), axis=1, power=power)


def _mb_s_flux(amplitudes, split_size=0.05):
    """Mean positive change from the previous frame; first frame starts from silence."""
    amplitudes = to_mono(amplitudes)
    mb = get_multibands(np.asarray(amplitudes, dtype=float), split_size)
    return np.array([
        np.maximum(np.diff(band, axis=0, prepend=np.zeros((1, *band.shape[1:]))), 0).sum(axis=1)
        / max(band.shape[1], 1)
        for band in mb
    ])


def get_mb_zcr(amplitudes, split_size=0.05, power=1.0):
    """Row sign-change rate. Magnitude spectra give zero; waveform ZCR needs signed samples."""
    amplitudes = to_mono(amplitudes)
    mb = get_multibands(amplitudes, split_size)
    return normalize(np.array([
        np.count_nonzero(np.diff(np.signbit(band), axis=1), axis=1)
        / max(band.shape[1] - 1, 1)
        for band in mb
    ]), axis=1, power=power)


def get_mb_mid_side_e(amplitudes, band_indices=None, n_bands=None, split_size=0.05, power=1.0, frames_per_group=1):
    """Normalized mean mid/side energy; magnitude input measures spectral energy, not stereo width."""
    amplitudes = np.asarray(amplitudes)
    if amplitudes.ndim != 3 or amplitudes.shape[-1] != 2:
        raise ValueError("Mid/side energy requires stereo data shaped (frames, bins, 2), not mono magnitudes")
    if band_indices is None:
        mb = get_multibands(amplitudes, split_size)
    else:
        # Use the same logarithmic band boundaries as the magnitude features.
        boundaries = (n_bands * get_splits(split_size)).astype(int)
        mb = np.split(amplitudes, np.searchsorted(band_indices, boundaries), axis=1)
    energy = to_mono(np.array([
        np.stack([
            (np.abs((band[..., 0] + band[..., 1]) * 0.5) ** 2).sum(axis=1),
            (np.abs((band[..., 0] - band[..., 1]) * 0.5) ** 2).sum(axis=1),
        ], axis=-1) / max(band.shape[1], 1)
        for band in mb
    ]))
    if frames_per_group < 1:
        raise ValueError("frames_per_group must be positive")
    energy = np.pad(energy, ((0, 0), (0, (-energy.shape[1]) % frames_per_group)))
    energy = energy.reshape(len(energy), -1, frames_per_group).mean(axis=2)
    return normalize(energy, axis=1, power=power)


def get_mb_crest_factor(amplitudes, split_size=0.05, power=1.0):
    """Peak/RMS of combined channel magnitudes across axis 1; silence gives zero."""
    amplitudes = to_mono(amplitudes)
    mb = get_multibands(np.asarray(amplitudes, dtype=float), split_size)
    result = []
    for band in mb:
        rms = np.sqrt((band ** 2).sum(axis=1) / max(band.shape[1], 1))
        peak = np.max(np.abs(band), axis=1, initial=0)
        result.append(np.divide(peak, rms, out=np.zeros_like(rms), where=rms != 0))
    return normalize(np.array(result), axis=1, power=power)
