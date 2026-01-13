"""
Quick start example - Get up and running with pylipsync in just a few lines.
"""

from pylipsync import PhonemeAnalyzer

analyzer = PhonemeAnalyzer()

segments = analyzer.extract_phoneme_segments("path/to/your/audio.mp3")

for segment in segments:
    print(f"{segment.start}-{segment.end} | Dominant Phoneme: {segment.dominant_phoneme.name}")
