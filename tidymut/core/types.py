# tidymut/types.py
from __future__ import annotations

from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .sequence import BaseSequence, MutationSet
    from .mutation import BaseMutation
    from ..cleaners.base_config import BaseCleanerConfig

SequenceType = TypeVar("SequenceType", bound="BaseSequence")
MutationType = TypeVar("MutationType", bound="BaseMutation")
MutationSetType = TypeVar("MutationSetType", bound="MutationSet")
CleanerConfigType = TypeVar("CleanerConfigType", bound="BaseCleanerConfig")
