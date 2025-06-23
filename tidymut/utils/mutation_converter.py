# tidymut/utils/mutation_converter.py

from ..core.mutation import (
    AminoAcidMutationSet,
    CodonMutationSet,
)
from ..core.types import MutationType, MutationSetType


def invert_mutation(mutation: MutationType) -> MutationType:
    """Helper function to invert a mutation"""
    mutation_type = type(mutation)
    return mutation_type(
        wild_type=mutation.mutant_type,
        position=mutation.position,
        mutant_type=mutation.wild_type,
        alphabet=mutation.alphabet,
        metadata=mutation.metadata,
    )


def invert_mutation_set(mutation_set: MutationSetType) -> MutationSetType:
    """Helper function to invert a mutation set"""
    mutation_set_type = type(mutation_set)
    inverted_mutations = [invert_mutation(mut) for mut in mutation_set.mutations]
    if mutation_set_type in (AminoAcidMutationSet, CodonMutationSet):
        return mutation_set_type(
            mutations=inverted_mutations,
            name=mutation_set.name,
            metadata=mutation_set.metadata,
        )

    return mutation_set_type(
        mutations=inverted_mutations,
        mutation_type=mutation_set.mutation_type,
        name=mutation_set.name,
        metadata=mutation_set.metadata,
    )
