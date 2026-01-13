import numpy as np
import librosa as lb


def downsample(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """Downsample audio to target sample rate.
    
    Args:
        audio: Input audio array.
        sample_rate: Current sample rate.
        target_sample_rate: Target sample rate.
    
    Returns:
        Downsampled audio array.
    """
    if sample_rate <= target_sample_rate:
        return audio.copy()
    
    resampled = lb.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
    return resampled.astype(np.float32)


def rms_volume(audio: np.ndarray) -> float:
    """Calculate RMS volume of audio.
    
    Args:
        audio: Input audio array.
    
    Returns:
        RMS volume value.
    """
    return np.sqrt(np.mean(audio ** 2))
