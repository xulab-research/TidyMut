# tidymut/__init__.py
"""
TidyMut: A Python package for cleaning protein mutation data
"""

__version__ = "0.1.0"

from .core.alphabet import DNAAlphabet, RNAAlphabet, ProteinAlphabet
from .core.sequence import ProteinSequence, DNASequence, RNASequence
from .core.mutation import (
    AminoAcidMutation,
    CodonMutation,
    AminoAcidMutationSet,
    CodonMutationSet,
)
