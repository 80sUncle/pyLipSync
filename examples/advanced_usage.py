"""
Advanced usage example demonstrating all configuration options and features.
"""

import librosa as lb
from pylipsync import PhonemeAnalyzer, CompareMethod

analyzer = PhonemeAnalyzer(
    # Comparison method for phoneme matching
    compare_method=CompareMethod.COSINE_SIMILARITY,  # Options: COSINE_SIMILARITY, L1_NORM, L2_NORM
    
    # Silence detection threshold (0.0 to 1.0)
    # Lower values = lower threshold needed for a segment to be considered silence
    silence_threshold=0.3,
    
    # Custom phoneme name for silence (if using custom templates)
    silence_phoneme="silence"
)

# Load directly from file path
segments = analyzer.extract_phoneme_segments(
    "path/to/your/audio.mp3",
    
    # Analysis window size in milliseconds
    # Larger = more accurate MFCC, but less temporal resolution
    window_size_ms=64.0,
    
    # Output frame rate (segments per second)
    # Higher FPS = more granular lip sync
    fps=60,
    
    # Return times in seconds vs. sample indices
    return_seconds=True,
    
    # Include the raw audio chunk that is analyzed in each segment
    return_audio=False
)

for segment in segments[:5]:
    print(f"\n({segment.start:.4f}s - {segment.end:.4f}s)")
    print(f"  Dominant phoneme: {segment.dominant_phoneme.name} ({segment.dominant_phoneme.target:.3f})")


# Load audio with librosa (or any audio library)
audio, sample_rate = lb.load("path/to/your/audio.mp3", sr=None)

segments = analyzer.extract_phoneme_segments(
    audio,
    sample_rate,  # Required when audio is a NumPy array
    window_size_ms=64.0,
    fps=60,
    return_seconds=False,  # Get sample indices instead of seconds
    return_audio=True      # Include audio chunks in segments
)

for i, segment in enumerate(segments[:5]):
    print(f"\nSegment {i}:")
    print(f"  Start sample: {segment.start}")
    print(f"  End sample: {segment.end}")
    print(f"  Audio chunk shape: {segment.audio.shape}")
    print(f"  Dominant phoneme: {segment.dominant_phoneme.name}")