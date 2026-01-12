import pytest
import numpy as np
from pylipsync import PhonemeAnalyzer

@pytest.fixture
def simple_sine_wave():
    """Generate a simple 440Hz sine wave (1 second, 16kHz)."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate

@pytest.fixture
def silence_audio():
    """Generate silence (1 second, 16kHz)."""
    sample_rate = 16000
    duration = 1.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    return audio, sample_rate

@pytest.fixture
def lipsync():
    """Create a LipSync instance."""
    return PhonemeAnalyzer()