from enum import Enum
import numpy as np


class CompareMethod(Enum):
    L1_NORM = "L1Norm"
    L2_NORM = "L2Norm"
    COSINE_SIMILARITY = "CosineSimilarity"

def _l1_score(mfcc: np.ndarray, template: np.ndarray) -> float:
    distance = np.mean(np.abs(mfcc - template))
    return 10.0 ** (-distance)

def _l2_score(mfcc: np.ndarray, template: np.ndarray) -> float:
    diff = mfcc - template
    distance = np.sqrt(np.mean(diff ** 2))
    return 10.0 ** (-distance)

def _cosine_score(mfcc: np.ndarray, template: np.ndarray) -> float:
    dot_product = np.dot(mfcc, template)
    norm_mfcc = np.linalg.norm(mfcc)
    norm_template = np.linalg.norm(template)
    
    if norm_mfcc == 0 or norm_template == 0:
        similarity = 0.0
    else:
        similarity = dot_product / (norm_mfcc * norm_template)
    
    similarity = max(similarity, 0.0)
    return similarity ** 100.0


_SCORE_FUNCTIONS = {
    CompareMethod.L1_NORM: _l1_score,
    CompareMethod.L2_NORM: _l2_score,
    CompareMethod.COSINE_SIMILARITY: _cosine_score,
}

def compute_similarity(mfcc: np.ndarray, template: np.ndarray, compare_method: CompareMethod) -> float:
    """Compute similarity score between MFCC and template.
    
    Args:
        mfcc: MFCC feature vector.
        template: Template MFCC feature vector.
        compare_method: Comparison method to use.
    
    Returns:
        Similarity score.
    """
    if compare_method not in _SCORE_FUNCTIONS:
        raise ValueError(f"Unimplemented comparison method: {compare_method}")
    return _SCORE_FUNCTIONS[compare_method](mfcc, template)
