"""MFCC-based phoneme analyzer for lip synchronization."""

import os
import json
import logging
import numpy as np
import librosa as lb
from collections import defaultdict

from .utils import downsample, rms_volume, compute_mfcc
from .types import Phoneme, PhonemeSegment
from .similarity import CompareMethod, compute_similarity

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


class PhonemeAnalyzer:
    """MFCC-based phoneme analyzer for lip synchronization.
    
    Compares input audio against pre-computed phoneme templates
    to determine phoneme targets for each audio segment.
    """
    AUDIO_EXTENSIONS = ("wav", "mp3", "ogg", "flac", "m4a", "wma", "aac", "aiff", "au", "raw", "pcm")
    TARGET_SAMPLE_RATE = 16000
    RANGE_HZ = 500
    MIN_VOLUME = -2.5
    MAX_VOLUME = -1.5
    EPSILON = 1e-10
    
    def __init__(
        self, 
        phoneme_templates_path: str = "phonemes/template.json",
        audio_templates_path: str = "phonemes/audio",
        compare_method: CompareMethod = CompareMethod.COSINE_SIMILARITY,
        silence_threshold: float = 0.3,
        silence_phoneme: str = "silence"
    ):
        """Initialize phoneme analyzer.
        
        Args:
            phoneme_templates_path: Path to phoneme templates file.
            audio_templates_path: Path to audio templates directory.
            compare_method: Similarity metric to use.
            silence_threshold: Threshold for silence detection (0-1).
            silence_phoneme: Name of the silence phoneme.
        """
        if silence_threshold < 0 or silence_threshold > 1:
            raise ValueError(
                f"Silence threshold must be between 0 and 1, got {silence_threshold}. "
                f"Use a value closer to 0 for higher silence detection and closer to 1 for lower silence detection."
            )

        self.phoneme_templates_path = os.path.join(MODULE_DIR, phoneme_templates_path)
        self.audio_templates_path = os.path.join(MODULE_DIR, audio_templates_path)
        self.compare_method = compare_method
        self.silence_threshold = silence_threshold
        self.silence_phoneme = silence_phoneme

        template_data = self._load_template()
        self.phoneme_mfccs: dict[str, list[list[float]]] = template_data["phonemes"]
        self.means: list[float] = template_data["normalization"]["means"]
        self.std_devs: list[float] = template_data["normalization"]["std_devs"]

    def _load_template(self) -> dict:
        try:
            template_data = self._read_template_file()
            logger.debug(f"Phoneme templates loaded from {self.phoneme_templates_path}")
        except FileNotFoundError:
            logger.info(f"Phoneme templates file not found at {self.phoneme_templates_path}, building templates...")
            template_data = self._generate_template()
            logger.info(f"Phoneme templates built and saved to {self.phoneme_templates_path}")
        return template_data

    def _read_template_file(self) -> dict:
        with open(self.phoneme_templates_path) as f:
            return json.load(f)

    def _get_phoneme_audio_folders(self) -> list[str]:
        folders = [
            os.path.join(self.audio_templates_path, name)
            for name in os.listdir(self.audio_templates_path)
            if os.path.isdir(os.path.join(self.audio_templates_path, name))
        ]
        if not folders:
            raise FileNotFoundError(f"No folders found within {self.audio_templates_path}!")
        return folders

    def _generate_template(self) -> dict:
        phoneme_mfccs = defaultdict(list)
        phoneme_audio_folders = self._get_phoneme_audio_folders()

        for folder in phoneme_audio_folders:
            phoneme = os.path.basename(folder)
            
            for file in os.listdir(folder):
                if not file.endswith(self.AUDIO_EXTENSIONS):
                    continue

                audio_file_path = os.path.join(folder, file)
                audio, _ = lb.load(audio_file_path, sr=self.TARGET_SAMPLE_RATE)

                mfcc, _ = self._extract_features(audio, self.TARGET_SAMPLE_RATE)
                phoneme_mfccs[phoneme].append(mfcc)

        if not phoneme_mfccs:
            raise FileNotFoundError(f"Could not create phoneme templates. No audio files were found!")

        means, std_devs = self._compute_normalization_stats(phoneme_mfccs)

        template_data = {
            "phonemes": dict(phoneme_mfccs),
            "normalization": {
                "means": means,
                "std_devs": std_devs
            }
        }

        with open(self.phoneme_templates_path, "w") as f:
            json.dump(template_data, f, indent=4)
        
        return template_data

    def _compute_normalization_stats(self, phoneme_mfccs: dict[str, list]) -> tuple[list, list]:
        all_mfccs = np.concatenate(list(phoneme_mfccs.values()))
        
        means = np.mean(all_mfccs, axis=0)
        std_devs = np.std(all_mfccs, axis=0)
        std_devs = np.where(std_devs < self.EPSILON, 1.0, std_devs)
        
        return means.tolist(), std_devs.tolist()

    def _compute_phoneme_targets(self, mfcc: np.ndarray) -> list[Phoneme]:
        means = np.array(self.means)
        std_devs = np.array(self.std_devs)
        normalized_mfcc = (mfcc - means) / std_devs
        
        phonemes = []
        for phoneme, template_mfccs in self.phoneme_mfccs.items():
            target = sum(
                compute_similarity(
                    normalized_mfcc,
                    (np.array(template_mfcc) - means) / std_devs,
                    self.compare_method
                )
                for template_mfcc in template_mfccs
            )
            phonemes.append(Phoneme(phoneme, target))
        
        return phonemes

    def _separate_silence_phoneme(self, phonemes: list[Phoneme]) -> tuple[list[Phoneme], Phoneme]:
        silence_phoneme = next((p for p in phonemes if p.name == self.silence_phoneme), None)
        
        if silence_phoneme is None:
            raise ValueError(f"Silence phoneme '{self.silence_phoneme}' not found")
        
        non_silence_phonemes = [p for p in phonemes if p.name != self.silence_phoneme]
        return non_silence_phonemes, silence_phoneme

    def _predict_phonemes(self, mfcc: np.ndarray, volume: float) -> list[Phoneme]:
        phonemes = self._compute_phoneme_targets(mfcc)
        
        phonemes, silence_phoneme = self._separate_silence_phoneme(phonemes)

        if self._silence_threshold_met(silence_phoneme, phonemes):
            phonemes = self._zero_out_targets(phonemes)
        else:
            phonemes = self._normalize_targets(phonemes)
            phonemes = self._apply_volume_weighting(phonemes, self._normalize_volume(volume))

        return phonemes

    def _silence_threshold_met(self, silence_phoneme: Phoneme, phonemes: list[Phoneme]) -> bool:
        total = silence_phoneme.target + sum(phoneme.target for phoneme in phonemes)
        return silence_phoneme.target / total >= self.silence_threshold

    def _normalize_volume(self, volume: float) -> float:
        if volume < self.EPSILON:
            return 0.0

        log_volume = np.log10(volume)
        normalized = (log_volume - self.MIN_VOLUME) / max(self.MAX_VOLUME - self.MIN_VOLUME, self.EPSILON)
        return float(np.clip(normalized, 0.0, 1.0))

    def _zero_out_targets(self, phonemes: list[Phoneme]) -> list[Phoneme]:
        for phoneme in phonemes:
            phoneme.target = 0.0
        return phonemes

    def _apply_volume_weighting(self, phonemes: list[Phoneme], volume: float) -> list[Phoneme]:
        for phoneme in phonemes:
            phoneme.target *= volume
        return phonemes

    def _normalize_targets(self, phonemes: list[Phoneme]) -> list[Phoneme]:
        total = sum(phoneme.target for phoneme in phonemes)
        if total > 0:
            for phoneme in phonemes:
                phoneme.target /= total
        return phonemes

    def _extract_features(self, audio: np.ndarray, sample_rate: int) -> tuple[list[float], float]:
        volume = rms_volume(audio)
        mfcc = compute_mfcc(audio, sample_rate, range_hz=self.RANGE_HZ)
        return mfcc, volume

    def _validate_extraction_params(
        self,
        audio: np.ndarray,
        sample_rate: int,
        window_size_ms: float,
        fps: int
    ) -> None:
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio data, got {audio.ndim}D")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if window_size_ms <= 0:
            raise ValueError(f"Window size must be positive, got {window_size_ms}")
        if fps <= 0:
            raise ValueError(f"FPS must be at least 1, got {fps}")

    def extract_phoneme_segments(
        self,
        audio: np.ndarray | str,
        sample_rate: int | None = None,
        *,
        window_size_ms: float = 64.0,
        fps: int = 60,
        return_audio: bool = False,
        return_seconds: bool = False
    ) -> list[PhonemeSegment]:
        """Extract phoneme segments from audio.
        
        Args:
            audio: 1D audio array.
            sample_rate: Sample rate of the audio.
            window_size_ms: Analysis window size in milliseconds.
            fps: Output frames per second.
            return_audio: Include audio data in each segment.
            return_seconds: Return times in seconds instead of samples.
        
        Returns:
            List of PhonemeSegment objects, one per frame.
        """
        if isinstance(audio, np.ndarray) and sample_rate is None:
            raise ValueError(f"Sample rate must be provided if audio is a numpy array, got {sample_rate}")
        
        if isinstance(audio, str):
            audio, sample_rate = lb.load(audio, sr=None)
        
        self._validate_extraction_params(audio, sample_rate, window_size_ms, fps)

        downsampled_audio = downsample(audio, sample_rate, self.TARGET_SAMPLE_RATE)
        window_size = int((window_size_ms / 1000) * self.TARGET_SAMPLE_RATE)
        
        if len(downsampled_audio) < window_size:
            raise ValueError(f"Downsampled audio too short: {len(downsampled_audio)} samples < {window_size} samples")
        
        hop_size = self.TARGET_SAMPLE_RATE // fps
        sample_rate_ratio = sample_rate / self.TARGET_SAMPLE_RATE
        
        segments = []
        for i in range(0, len(downsampled_audio) - window_size + 1, hop_size):
            downsampled_audio_chunk = downsampled_audio[i: i + window_size]
            
            original_audio_start = int(i * sample_rate_ratio)
            original_audio_end = int((i + hop_size) * sample_rate_ratio)
            original_audio_chunk = audio[original_audio_start: original_audio_end]
            
            mfcc, volume = self._extract_features(downsampled_audio_chunk, self.TARGET_SAMPLE_RATE)
            phonemes = self._predict_phonemes(mfcc, volume)
            segments.append(
                PhonemeSegment(
                    phonemes,
                    original_audio_start / sample_rate if return_seconds else original_audio_start, 
                    original_audio_end / sample_rate if return_seconds else original_audio_end, 
                    original_audio_chunk if return_audio else None,
                )
            )
        
        return segments
