# tidymut/types.py

from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .sequence import BaseSequence, MutationSet
    from .mutation import BaseMutation

SequenceType = TypeVar("SequenceType", bound="BaseSequence")
MutationType = TypeVar("MutationType", bound="BaseMutation")
MutationSetType = TypeVar("MutationSetType", bound="MutationSet")
