"""
Quick start example showing how to use pylipsync to analyze audio.

Demonstrates both methods of loading audio:
1. Direct file path (simplest)
2. Pre-loaded numpy array (useful for preprocessing)
"""

from pylipsync import PhonemeAnalyzer, CompareMethod

# Initialize the analyzer
analyzer = PhonemeAnalyzer(
    compare_method=CompareMethod.COSINE_SIMILARITY  # Options: L1_NORM, L2_NORM, COSINE_SIMILARITY
)

# Method 1: Pass audio file path directly (simplest)
print("Method 1: Direct file path")
segments = analyzer.extract_phoneme_segments(
    "path/to/your/audio.mp3",
    window_size_ms=64.0,
    fps=60,
    return_seconds=True  # Return times in seconds instead of sample indices
)

for segment in segments:
    most_prominent_phoneme = segment.most_prominent_phoneme() if not segment.is_silence() else None
    print(f"({segment.start:.4f}-{segment.end:.4f}) | Most Prominent Phoneme: {most_prominent_phoneme}")


# Method 2: Pre-load audio
print("\nMethod 2: Pre-loaded numpy array")
import librosa as lb

audio, sr = lb.load("path/to/your/audio.mp3", sr=None)

segments = analyzer.extract_phoneme_segments(
    audio,
    sr,
    window_size_ms=64.0,
    fps=60,
    return_seconds=True
)

for segment in segments:
    most_prominent_phoneme = segment.most_prominent_phoneme() if not segment.is_silence() else None
    print(f"({segment.start:.4f}-{segment.end:.4f}) | Most Prominent Phoneme: {most_prominent_phoneme}")
