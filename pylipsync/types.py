import numpy as np
from dataclasses import dataclass


@dataclass
class Phoneme:
    """Represents a single phoneme with a target confidence score.
    
    Attributes:
        name: The phoneme identifier (e.g., 'aa', 'ee', 'ih').
        target: Confidence score for this phoneme (0.0 to 1.0).
    """
    name: str
    target: float

    def to_dict(self) -> dict:
        return {"name": self.name, "target": self.target}

@dataclass
class PhonemeSegment:
    """Represents a single audio segment with phoneme analysis results.
    
    Attributes:
        phonemes: List of detected phonemes with confidence scores.
        start: Start time/position (seconds if return_seconds=True, samples otherwise).
        end: End time/position (seconds if return_seconds=True, samples otherwise).
        audio: Optional audio data for this segment (only included if return_audio=True).
    """
    phonemes: list[Phoneme]
    start: int | float
    end: int | float
    audio: np.ndarray | None = None

    def to_dict(self):
        segment = {
            "phonemes": [phoneme.to_dict() for phoneme in self.phonemes],
            "start": self.start,
            "end": self.end,
        }
        
        if self.audio is not None:
            segment["audio"] = self.audio.tobytes()

        return segment

    @property
    def dominant_phoneme(self) -> Phoneme:
        return max(self.phonemes, key=lambda phoneme: phoneme.target)