# tidymut/utils/dataset_builders.py

"""
Functions are used in tidymut.cleaners.basic_cleaners.convert_to_mutation_dataset_format()
>>> # format 1:
>>> pd.DataFrame({
...     'name': ['prot1', 'prot1', 'prot1', 'prot2', 'prot2'],
...     'mut_info': ['A0S,Q1D', 'C2D', 'WT', 'E0F', 'WT'],
...     'mut_seq': ['SDCDEF', 'AQDDEF', 'AQCDEF', 'FGHIGHK', 'EGHIGHK'],
...     'score': [1.5, 2.0, 0.0, 3.0, 0.0]
... })
>>>
>>> # format 2:
>>> df2 = pd.DataFrame({
...     'name': ['prot1', 'prot1', 'prot2'],
...     'sequence': ['AKCDEF', 'AKCDEF', 'FEGHIS'],
...     'mut_info': ['A0K,C2D', 'Q1P', 'E1F'],
...     'score': [1.5, 2.0, 3.0],
...     'mut_seq': ['KKDDEF', 'APCDEF', 'FFGHIS']
... })
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING

from ..core.mutation import MutationSet

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Type, Tuple, Union

    from ..core.sequence import ProteinSequence, DNASequence, RNASequence

__all__ = ["convert_format_1", "convert_format_2"]


def __dir__() -> List[str]:
    return __all__


def convert_format_1(
    df: pd.DataFrame,
    name_column: str,
    mutation_column: str,
    mutated_sequence_column: str,
    score_column: str,
    include_wild_type: bool,
    mutation_set_prefix: str,
    is_zero_based: bool,
    additional_metadata: Optional[Dict[str, Any]],
    sequence_class: Type[Union[ProteinSequence, DNASequence, RNASequence]],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Convert Format 1 (with WT rows) to mutation dataset format."""

    input_df = df.copy()

    # Extract reference sequences from WT rows
    wt_rows = input_df[input_df[mutation_column] == "WT"]
    if wt_rows.empty:
        raise ValueError("No wild-type (WT) entries found in the dataset")

    reference_sequences = {}
    for _, row in wt_rows.iterrows():
        name = row[name_column]
        sequence = row[
            mutated_sequence_column
        ]  # For WT rows, this is the wild-type sequence
        reference_sequences[name] = sequence_class(sequence)

    # Filter out wild-type entries if requested
    if not include_wild_type:
        input_df = input_df[input_df[mutation_column] != "WT"].copy()

    if input_df.empty:
        raise ValueError("No mutation data remaining after filtering")

    # Process mutations (now supporting multi-mutations)
    output_rows = []
    total_rows = len(input_df)
    for idx, row in tqdm(enumerate(input_df.itertuples()), total=total_rows):
        mut_info = getattr(row, mutation_column)
        name = getattr(row, name_column)
        score = getattr(row, score_column)

        # Skip wild-type if it somehow made it through filtering
        if mut_info == "WT":
            continue

        # Parse mutations (single or multiple)
        try:
            mutation_data_list = _parse_mutations_string(mut_info, is_zero_based)
        except ValueError as e:
            raise ValueError(f"Cannot parse mutation '{mut_info}' in row {idx}: {e}")

        # Create one output row per individual mutation within the set
        mutation_set_id = f"{mutation_set_prefix}_{idx + 1}"
        mutation_set_name = f"{name}_{mut_info}"

        for mutation_data in mutation_data_list:
            output_row = _create_output_row_from_mutation_data(
                mutation_set_id,
                mutation_set_name,
                mut_info,
                name,
                score,
                mutation_data,
                additional_metadata,
            )
            output_rows.append(output_row)

    output_df = pd.DataFrame(output_rows)
    return output_df, reference_sequences


def convert_format_2(
    df: pd.DataFrame,
    name_column: str,
    mutation_column: str,
    sequence_column: str,
    score_column: str,
    mutation_set_prefix: str,
    is_zero_based: bool,
    additional_metadata: Optional[Dict[str, Any]],
    sequence_class: Type[Union[ProteinSequence, DNASequence, RNASequence]],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Convert Format 2 (with sequence column) to mutation dataset format."""

    input_df = df.copy()

    # Extract reference sequences from sequence column
    reference_sequences = {}
    for name, group in tqdm(input_df.groupby(name_column)):
        sequences = group[sequence_column].unique()
        if len(sequences) > 1:
            raise ValueError(
                f"Multiple different sequences found for protein '{name}': {sequences}"
            )
        reference_sequences[name] = sequence_class(sequences[0])

    # Process mutations (now supporting multi-mutations)
    output_rows = []
    total_rows = len(input_df)
    for idx, row in tqdm(enumerate(input_df.itertuples()), total=total_rows):
        mut_info = getattr(row, mutation_column)
        name = getattr(row, name_column)
        score = getattr(row, score_column)

        # Parse mutations (single or multiple)
        try:
            mutation_data_list = _parse_mutations_string(mut_info, is_zero_based)
        except ValueError as e:
            raise ValueError(f"Cannot parse mutation '{mut_info}' in row {idx}: {e}")

        # Create one output row per individual mutation within the set
        mutation_set_id = f"{mutation_set_prefix}_{idx + 1}"
        mutation_set_name = f"{name}_{mut_info}"

        for mutation_data in mutation_data_list:
            output_row = _create_output_row_from_mutation_data(
                mutation_set_id,
                mutation_set_name,
                mut_info,
                name,
                score,
                mutation_data,
                additional_metadata,
            )
            output_rows.append(output_row)

    output_df = pd.DataFrame(output_rows)
    return output_df, reference_sequences


def _create_output_row_from_mutation_data(
    mutation_set_id: str,
    mutation_set_name: str,
    original_mutation_string: str,
    name: str,
    score: float,
    mutation_data: Dict[str, Any],
    additional_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Create a single output row from mutation data.

    Parameters
    ----------
    mutation_set_id : str
        ID for the mutation set
    mutation_set_name : str
        Name for the mutation set
    original_mutation_string : str
        Original mutation string (may contain multiple mutations)
    name : str
        Protein/sequence name
    score : float
        Score associated with the mutation set
    mutation_data : Dict[str, Any]
        Data for a single mutation
    additional_metadata : Optional[Dict[str, Any]]
        Additional metadata for the mutation set

    Returns
    -------
    Dict[str, Any]
        Row data for the output DataFrame
    """
    output_row = {
        "mutation_set_id": mutation_set_id,
        "reference_id": name,
        "mutation_string": mutation_data["mutation_string"],  # Individual mutation
        "position": mutation_data["position"],
        "mutation_type": "amino_acid",
        "wild_amino_acid": mutation_data["wild_aa"],
        "mutant_amino_acid": mutation_data["mutant_aa"],
        "mutation_set_name": mutation_set_name,
        "label": score,
        "set_original_mutation_string": original_mutation_string,  # Store original string
    }

    # Add additional metadata if provided
    if additional_metadata:
        for key, value in additional_metadata.items():
            output_row[f"set_{key}"] = value

    return output_row


def _parse_mutations_string(
    mutation_string: str, is_zero_based: bool
) -> list[Dict[str, Any]]:
    """
    Parse a mutation string that may contain single or multiple mutations.

    This function can handle:
    - Single mutations: 'A0S'
    - Multiple mutations: 'A0S,Q1D' or 'A0S;Q1D'

    Uses MutationSet.from_string to parse complex mutation strings and
    falls back to simple parsing for basic cases.

    Parameters
    ----------
    mutation_string : str
        Mutation string(s) to parse

    is_zero_based : bool
        Whether origin mutation positions are zero-based

    Returns
    -------
    list[Dict[str, Any]]
        List of mutation data dictionaries, each containing:
        - 'wild_aa': wild-type amino acid
        - 'position': position (0-based)
        - 'mutant_aa': mutant amino acid
        - 'mutation_string': individual mutation string

    Raises
    ------
    ValueError
        If the mutation string cannot be parsed
    """
    mutation_string = mutation_string.strip()

    # Use MutationSet.from_string to parse complex mutation strings
    mutation_set = MutationSet.from_string(mutation_string, is_zero_based=is_zero_based)

    mutation_data_list = []
    for mutation in mutation_set.mutations:
        # Extract information from the mutation object
        if (
            hasattr(mutation, "wild_type")
            and hasattr(mutation, "position")
            and hasattr(mutation, "mutant_type")
        ):
            mutation_data = {
                "wild_aa": mutation.wild_type,
                "position": mutation.position,
                "mutant_aa": mutation.mutant_type,
                "mutation_string": str(mutation),  # Individual mutation string
            }
            mutation_data_list.append(mutation_data)
        else:
            raise ValueError(
                f"Mutation object does not have expected attributes: {mutation}"
            )

    if not mutation_data_list:
        raise ValueError("No valid mutations found in mutation set")

    return mutation_data_list
