"""DST-Fusion: Dempster-Shafer evidence fusion for type 2 diabetes screening."""
from .classifier import DSTFusionClassifier
from . import core, simulate, evaluation

__all__ = ["DSTFusionClassifier", "core", "simulate", "evaluation"]
__version__ = "1.0.0"
