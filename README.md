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

Get started with just a few lines of code:

```python
from pylipsync import PhonemeAnalyzer

analyzer = PhonemeAnalyzer()

segments = analyzer.extract_phoneme_segments("path/to/your/audio.mp3")

for segment in segments:
    print(f"{segment.start}-{segment.end}: {segment.dominant_phoneme.name}")
```

### Advanced Usage

For more control over the analysis, you can customize the analyzer and extraction parameters:

```python
from pylipsync import PhonemeAnalyzer, CompareMethod
import librosa as lb

# Initialize with custom settings
analyzer = PhonemeAnalyzer(
    compare_method=CompareMethod.COSINE_SIMILARITY,  # L1_NORM, L2_NORM, COSINE_SIMILARITY
    silence_threshold=0.3
)

# Method 1: Pass file path directly
segments = analyzer.extract_phoneme_segments(
    "path/to/your/audio.mp3",
    window_size_ms=64.0,    # Analysis window size
    fps=60,                 # Output frame rate
    return_seconds=True     # Return times in seconds
)

# Method 2: Pre-load audio as NumPy array
audio, sr = lb.load("path/to/your/audio.mp3", sr=None)
segments = analyzer.extract_phoneme_segments(
    audio,
    sr,                     # Required when passing NumPy array
    window_size_ms=64.0,
    fps=60,
    return_audio=True       # Include audio chunk in each segment
)

for segment in segments:
    print(f"{segment.start}-{segment.end} | Dominant Phoneme: {segment.dominant_phoneme.name}")
```

See [`examples/advanced_usage.py`](examples/advanced_usage.py) for a complete guide with all configuration options.

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
