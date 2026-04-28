"""
Generative models for synthetic UV-Vis spectrum generation.
"""

from .base import BaseGenerativeModel
from .cgan import CGAN

__all__ = [
    "BaseGenerativeModel",
    "CGAN",
]
