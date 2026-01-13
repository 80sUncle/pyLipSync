import pytest
import numpy as np
import librosa as lb
from pathlib import Path
from pylipsync import PhonemeAnalyzer, CompareMethod
import pylipsync


PACKAGE_DIR = Path(pylipsync.__file__).parent

class TestAnalyzerValidation:
    def test_invalid_silence_threshold(self):
        with pytest.raises(ValueError):
            PhonemeAnalyzer(silence_threshold=1.5)
        with pytest.raises(ValueError):
            PhonemeAnalyzer(silence_threshold=-0.1)
    
    def test_invalid_sample_rate(self, lipsync: PhonemeAnalyzer, simple_sine_wave: tuple[np.ndarray, int]):
        audio, _ = simple_sine_wave
        with pytest.raises(ValueError):
            lipsync.extract_phoneme_segments(audio, -1)
        with pytest.raises(ValueError):
            lipsync.extract_phoneme_segments(audio, 0)
    
    def test_invalid_audio_dimensions(self, lipsync: PhonemeAnalyzer):
        audio_2d = np.zeros((2, 1000))
        with pytest.raises(ValueError):
            lipsync.extract_phoneme_segments(audio_2d, 16000)
    
    def test_audio_too_short_after_downsampling(self, lipsync: PhonemeAnalyzer):
        sample_rate = 16000
        duration_ms = 10
        num_samples = int(sample_rate * duration_ms / 1000)
        audio = np.random.randn(num_samples)
        
        with pytest.raises(ValueError):
            lipsync.extract_phoneme_segments(audio, sample_rate, window_size_ms=64.0)


class TestAnalyzerProcessing:
    def test_process_audio_segments(self, lipsync: PhonemeAnalyzer, simple_sine_wave: tuple[np.ndarray, int]):
        audio, sr = simple_sine_wave
        segments = lipsync.extract_phoneme_segments(audio, sr)
        
        assert len(segments) > 0
        assert all(len(seg.phonemes) > 0 for seg in segments)
        assert all(seg.start >= 0 for seg in segments)
        assert all(seg.end > seg.start for seg in segments)
    
    def test_silence_detection(self, lipsync: PhonemeAnalyzer, silence_audio: tuple[np.ndarray, int]):
        audio, sr = silence_audio
        segments = lipsync.extract_phoneme_segments(audio, sr)
        
        silence_count = sum(1 for seg in segments if seg.dominant_phoneme.name == "silence")
        assert silence_count > len(segments) * 0.8
    
    def test_comparison_methods_all_work(self, simple_sine_wave: tuple[np.ndarray, int]):
        audio, sr = simple_sine_wave
        
        compare_methods = [CompareMethod.L1_NORM, CompareMethod.L2_NORM, CompareMethod.COSINE_SIMILARITY]
        for method in compare_methods:
            lipsync = PhonemeAnalyzer(compare_method=method)
            segments = lipsync.extract_phoneme_segments(audio, sr)
            assert len(segments) > 0
    
    def test_return_seconds_format(self, lipsync: PhonemeAnalyzer, simple_sine_wave: tuple[np.ndarray, int]):
        audio, sr = simple_sine_wave
        segments = lipsync.extract_phoneme_segments(audio, sr, return_seconds=True)
        
        assert len(segments) > 0

        for seg in segments:
            assert isinstance(seg.start, float)
            assert isinstance(seg.end, float)
            assert seg.start >= 0.0
            assert seg.end > seg.start
    
    def test_return_audio_included(self, lipsync: PhonemeAnalyzer, simple_sine_wave: tuple[np.ndarray, int]):
        audio, sr = simple_sine_wave
        segments = lipsync.extract_phoneme_segments(audio, sr, return_audio=True)
        
        assert len(segments) > 0

        for seg in segments:
            assert seg.audio is not None
            assert isinstance(seg.audio, np.ndarray)
            assert len(seg.audio) > 0
    
    def test_path_based_loading(self, lipsync: PhonemeAnalyzer):
        audio_path = PACKAGE_DIR / "phonemes" / "audio" / "aa" / "A_female.mp3"
        assert audio_path.exists(), f"Test audio file not found: {audio_path}"
        
        segments = lipsync.extract_phoneme_segments(str(audio_path))
        
        assert len(segments) > 0
        assert all(len(seg.phonemes) > 0 for seg in segments)
        assert all(seg.start >= 0 for seg in segments)
        assert all(seg.end > seg.start for seg in segments)


class TestAnalyzerIntegration:
    @pytest.mark.parametrize("phoneme,audio_file", [
        ("aa", "A_female.mp3"),
        ("ee", "E_female.mp3"),
        ("ih", "I_female.mp3"),
        ("oh", "O_female.mp3"),
        ("ou", "U_female.mp3"),
        ("silence", "silence.mp3"),
    ])
    def test_phoneme_detection(self, lipsync: PhonemeAnalyzer, phoneme: str, audio_file: str):
        """Test that each phoneme is correctly detected from its audio sample."""
        audio_path = PACKAGE_DIR / "phonemes" / "audio" / phoneme / audio_file
        
        assert audio_path.exists(), f"Audio file not found: {audio_path}"
        
        audio, sr = lb.load(str(audio_path), sr=None)
        segments = lipsync.extract_phoneme_segments(audio, sr)
        
        assert len(segments) > 0
        
        phoneme_counts = {}
        for seg in segments:
            name = seg.dominant_phoneme.name
            phoneme_counts[name] = phoneme_counts.get(name, 0) + 1
        
        most_common = max(phoneme_counts, key=phoneme_counts.get)
        assert most_common == phoneme, f"Expected '{phoneme}', got '{most_common}'"