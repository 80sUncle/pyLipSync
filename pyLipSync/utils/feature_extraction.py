"""MFCC feature extraction pipeline.

Implemented using: https://github.com/hecomi/uLipSync/blob/main/Assets/uLipSync/Runtime/Core/Algorithm.cs
"""

import numpy as np
import scipy.signal


def _low_pass_filter(audio_data: np.ndarray, sample_rate: int, cutoff: float, range_hz: float) -> np.ndarray:
    cutoff_norm = (cutoff - range_hz) / sample_rate
    range_norm = range_hz / sample_rate
    
    n = int(np.round(3.1 / range_norm))
    if (n + 1) % 2 == 0:
        n += 1
    
    b = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x = i - (n - 1) / 2.0
        ang = 2.0 * np.pi * cutoff_norm * x
        if abs(ang) < 1e-10:
            b[i] = 2.0 * cutoff_norm
        else:
            b[i] = 2.0 * cutoff_norm * np.sin(ang) / ang
    
    filtered = scipy.signal.lfilter(b, 1.0, audio_data)
    return filtered.astype(np.float32)


def _pre_emphasis(data: np.ndarray, p: float = 0.97) -> np.ndarray:
    emphasized = data.copy().astype(np.float32)
    emphasized[1:] = data[1:] - p * data[:-1]
    return emphasized


def _hamming_window(array: np.ndarray) -> np.ndarray:
    window = np.hamming(len(array)).astype(np.float32)
    return (array * window).astype(np.float32)


def _normalize_array(array: np.ndarray, value: float = 1.0) -> np.ndarray:
    max_val = np.max(np.abs(array))
    
    if max_val < np.finfo(float).eps:
        return array.astype(np.float32)
    
    return (array * (value / max_val)).astype(np.float32)


def _zero_padding(data: np.ndarray) -> np.ndarray:
    n = len(data)
    padded = np.zeros(n * 2, dtype=np.float32)
    padded[n//2:n//2 + n] = data
    return padded


def _fft_magnitude(data: np.ndarray) -> np.ndarray:
    fft_result = np.fft.fft(data)
    return np.abs(fft_result).astype(np.float32)


def _to_mel(hz: float, slaney: bool = False) -> float:
    a = 2595.0 if slaney else 1127.0
    return a * np.log(hz / 700.0 + 1.0)


def _to_hz(mel: float, slaney: bool = False) -> float:
    a = 2595.0 if slaney else 1127.0
    return 700.0 * (np.exp(mel / a) - 1.0)


def _mel_filter_bank(spectrum: np.ndarray, sample_rate: int, mel_div: int) -> np.ndarray:
    f_max = sample_rate / 2
    mel_max = _to_mel(f_max)
    n_max = len(spectrum) // 2
    df = f_max / n_max
    d_mel = mel_max / (mel_div + 1)
    
    mel_points = np.arange(mel_div + 2) * d_mel
    f_points = np.array([_to_hz(mel) for mel in mel_points])
    
    mel_spectrum = np.zeros(mel_div, dtype=np.float32)
    
    for n in range(mel_div):
        i_begin = int(np.ceil(f_points[n] / df))
        i_center = int(np.round(f_points[n + 1] / df))
        i_end = int(np.floor(f_points[n + 2] / df))
        
        if i_end > i_begin:
            indices = np.arange(i_begin + 1, i_end + 1)
            frequencies = df * indices
            
            left_mask = indices < i_center
            right_mask = indices >= i_center
            
            weights = np.zeros_like(frequencies)
            if np.any(left_mask):
                weights[left_mask] = (frequencies[left_mask] - f_points[n]) / (f_points[n + 1] - f_points[n])
            if np.any(right_mask):
                weights[right_mask] = (f_points[n + 2] - frequencies[right_mask]) / (f_points[n + 2] - f_points[n + 1])
            
            weights /= (f_points[n + 2] - f_points[n]) * 0.5
            mel_spectrum[n] = np.sum(weights * spectrum[indices])
    
    return mel_spectrum


def _power_to_db(array: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(array, 1e-10))).astype(np.float32)


def _dct(spectrum: np.ndarray) -> np.ndarray:
    n = len(spectrum)
    cepstrum = np.zeros(n, dtype=np.float32)
    a = np.pi / n
    
    for i in range(n):
        j_vals = np.arange(n)
        angles = (j_vals + 0.5) * i * a
        cepstrum[i] = np.sum(spectrum * np.cos(angles))
    
    return cepstrum


def compute_mfcc(
    audio: np.ndarray,
    sample_rate: int,
    range_hz: int = 500,
    mel_channels: int = 26,
    mfcc_num: int = 12
) -> list[float]:
    """Compute MFCC features from audio.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate of input audio.
        range_hz: Frequency range for low-pass filter.
        mel_channels: Number of mel filter bank channels.
        mfcc_num: Number of MFCC coefficients to return.
    
    Returns:
        List of MFCC coefficients.
    """
    cutoff = sample_rate / 2
    filtered = _low_pass_filter(audio, sample_rate, cutoff, range_hz)
    emphasized = _pre_emphasis(filtered, 0.97)
    windowed = _hamming_window(emphasized)
    normalized = _normalize_array(windowed, 1.0)
    padded = _zero_padding(normalized)
    spectrum = _fft_magnitude(padded)
    mel_spectrum = _mel_filter_bank(spectrum, sample_rate, mel_channels)
    mel_db = _power_to_db(mel_spectrum)
    mel_cepstrum = _dct(mel_db)
    
    return mel_cepstrum[1: mfcc_num + 1].tolist()
