"""
preprocess.algorithms

F0 extraction algorithm registry.

Usage:
    from preprocess.algorithms import get_algorithm
    cls  = get_algorithm('RMVPE')
    algo = cls(sample_rate=22050, hop_size=256, fmin=50.0, fmax=800.0)
"""

from .base import ContinuousPitchAlgorithm, PitchAlgorithm, ThresholdPitchAlgorithm
from .rmvpe import RMVPEPitchAlgorithm

_REGISTRY: dict[str, type] = {
    'RMVPE':         RMVPEPitchAlgorithm,
}


def get_algorithm(name: str) -> type:
    """
    Return the algorithm class for the given name.

    Available names: 'RMVPE'
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown algorithm: {name!r}. "
            f"Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


__all__ = [
    'get_algorithm',
    'PitchAlgorithm',
    'ContinuousPitchAlgorithm',
    'ThresholdPitchAlgorithm',
    'RMVPEPitchAlgorithm',
]
