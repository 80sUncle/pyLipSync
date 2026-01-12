# pylipsync

A Python implementation of [Hecomi's uLipSync](https://github.com/hecomi/uLipSync) for audio-based lip sync analysis. This library analyzes audio and determines phoneme targets for lip synchronization in real-time applications.

## Installation

### Install from PyPI

```bash
pip install pylipsync
```

### Install from Local Clone

Alternatively, clone the repository and install:

```bash
git clone https://github.com/spava002/pyLipSync.git
cd pyLipSync
pip install -e .
```

## Quick Start

The library comes with built-in audio templates for common phonemes, so you can start using it immediately:

```python
from pylipsync import PhonemeAnalyzer, CompareMethod

# Initialize PhonemeAnalyzer - works out of the box with default templates
analyzer = PhonemeAnalyzer(
    compare_method=CompareMethod.COSINE_SIMILARITY  # Options: L1_NORM, L2_NORM, COSINE_SIMILARITY
)

# Method 1: Pass audio file path directly (simplest)
segments = analyzer.extract_phoneme_segments(
    "path/to/your/audio.mp3",
    window_size_ms=64.0,    # Window size in milliseconds
    fps=60,                 # Frames per second for output
    return_seconds=True     # Return times in seconds (default: False = sample indices)
)

# Get the most prominent phoneme for each segment
for segment in segments:
    most_prominent_phoneme = segment.most_prominent_phoneme() if not segment.is_silence() else None
    print(f"({segment.start:.4f}-{segment.end:.4f}) | Most Prominent Phoneme: {most_prominent_phoneme}")
```

**Alternative: Pre-load audio**:

```python
import librosa as lb

# Method 2: Load audio first
audio, sr = lb.load("path/to/your/audio.mp3", sr=None)

segments = analyzer.extract_phoneme_segments(
    audio,
    sr,
    window_size_ms=64.0,
    fps=60,
    return_audio=True       # Include audio chunk in each segment (default: False)
)

for segment in segments:
    print(f"Segment audio shape: {segment.audio.shape if segment.audio is not None else 'None'}")
```

## Default Phonemes

The library includes pre-configured phoneme templates for:
- `aa` - "A" sounds
- `ee` - "E" sounds
- `ih` - "I" sounds
- `oh` - "O" sounds
- `ou` - "U" sounds
- `silence` - silence/no speech

These templates are ready to use without any additional setup.

### Adding New Phonemes

To add additional phonemes (e.g., consonants like "th", "sh", "f"):

1. Create a folder with all your phoneme names (or expand off the existing phonemes/audio/ folder)
   ```
   phonemes/audio/
   ├── aa/
   ├── ee/
   ├── th/          # New phoneme!
   │   └── th_sound.mp3
   └── sh/          # Another new one!
       └── sh_sound.mp3
   ```

2. Add audio samples to each folder (`.mp3`, `.wav`, `.ogg`, `.flac`, etc.)

3. Use your custom templates:
   ```python
   analyzer = PhonemeAnalyzer(
       audio_templates_path="/path/to/my_custom_audio"  # Not necessary if expanding within phonemes/audio/
   )
   ```

**Note:** The folder name becomes the phoneme identifier in the output.

## How It Works

1. **Template Loading**: The library loads pre-computed MFCC templates from `phonemes/template.json`
2. **Audio Processing**: Input audio is processed in overlapping windows using MFCC extraction
3. **Phoneme Matching**: Each segment is compared against all phoneme templates using the selected comparison method
4. **Target Calculation**: Returns normalized confidence scores (0-1) for each phoneme per segment
5. **Silence Detection**: Segments below the silence threshold have all phoneme targets set to 0

## Credits

This is a Python implementation of [uLipSync](https://github.com/hecomi/uLipSync) by Hecomi.

## License

MIT License - see [LICENSE](LICENSE) file for details.
