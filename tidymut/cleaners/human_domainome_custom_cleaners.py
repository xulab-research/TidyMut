# cleaners/human_domainome_custom_cleaners.py
from __future__ import annotations

import pandas as pd
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import TYPE_CHECKING

from ..core.pipeline import pipeline_step, multiout_step

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

__all__ = [
    "process_domain_positions",
    "add_sequences_to_dataset",
    "extract_domain_sequences",
]


def __dir__() -> List[str]:
    return __all__


@multiout_step(main="success", failed="failed")
def process_domain_positions(
    dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process domain position information from PFAM entries

    This function extracts position information from PFAM entries and calculates
    relative mutation positions. It handles parsing errors by separating failed records.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset with PFAM_entry column containing position information

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (successful_dataset, failed_dataset) - datasets with and without parsing errors

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'PFAM_entry': ['PF00001/10-100', 'PF00002/20-200', 'invalid_entry'],
    ...     'pos': [15, 25, 30],
    ...     'wt_aa': ['A', 'C', 'D'],
    ...     'mut_aa': ['K', 'Y', 'E']
    ... })
    >>> successful, failed = process_domain_positions(df)
    >>> print(len(successful))  # Should be 2
    2
    >>> print(len(failed))  # Should be 1
    1
    """
    tqdm.write("Extracting peptide position information...")

    # Create a copy to avoid modifying the original
    result_dataset = dataset.copy()
    result_dataset["error_message"] = None

    # Extract start and end positions from PFAM_entry (format: "PFAM_ENTRY/start-end")
    try:
        position_info = result_dataset["PFAM_entry"].str.extract(r"/(\d+)-(\d+)")

        # Track which entries failed to parse
        parse_failed = position_info.isnull().any(axis=1)
        result_dataset.loc[parse_failed, "error_message"] = (
            "Failed to parse PFAM_entry position information"
        )

        # Process successful entries
        success_mask = ~parse_failed

        if success_mask.any():
            result_dataset.loc[success_mask, "start_pos"] = (
                position_info.loc[success_mask, 0].astype(int) - 1
            )  # Convert to 0-based
            result_dataset.loc[success_mask, "end_pos"] = position_info.loc[
                success_mask, 1
            ].astype(int)

            # Convert absolute position to 0-based
            result_dataset.loc[success_mask, "pos"] = (
                result_dataset.loc[success_mask, "pos"] - 1
            )

            # Calculate relative position within the domain
            result_dataset.loc[success_mask, "mut_rel_pos"] = (
                result_dataset.loc[success_mask, "pos"]
                - result_dataset.loc[success_mask, "start_pos"]
            )

            # Generate mutation info using relative position
            result_dataset.loc[success_mask, "mut_info"] = (
                result_dataset.loc[success_mask, "wt_aa"]
                + result_dataset.loc[success_mask, "mut_rel_pos"]
                .astype(int)
                .astype(str)
                + result_dataset.loc[success_mask, "mut_aa"]
            )

    except Exception as e:
        # If something goes wrong, mark all as failed
        result_dataset["error_message"] = f"Error processing positions: {str(e)}"
        success_mask = pd.Series([False] * len(result_dataset))

    # Separate successful and failed datasets
    successful_dataset = result_dataset[success_mask].drop(columns=["error_message"])
    failed_dataset = result_dataset[~success_mask]

    tqdm.write(
        f"Position processing: {len(successful_dataset)} successful, {len(failed_dataset)} failed"
    )

    return successful_dataset, failed_dataset


@multiout_step(main="success", failed="failed")
def add_sequences_to_dataset(
    dataset: pd.DataFrame, sequence_dict: Dict[str, str], name_column: str = "name"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add full wild-type sequences to the dataset from sequence dictionary

    This function maps sequences from a dictionary to the dataset. Records without
    matching sequences are separated into the failed dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset containing protein names
    sequence_dict : Dict[str, str]
        Mapping from protein name to full wild-type sequence
    name_column : str, default='name'
        Column name containing protein identifiers

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (successful_dataset, failed_dataset) - datasets with and without sequences

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'name': ['prot1', 'prot2', 'prot3'],
    ...     'score': [1.0, 2.0, 3.0]
    ... })
    >>> seq_dict = {'prot1': 'AKCD', 'prot2': 'EFGH'}
    >>> successful, failed = add_sequences_to_dataset(df, seq_dict)
    >>> print(len(successful))  # Should be 2
    2
    >>> print(len(failed))  # Should be 1
    1
    """
    tqdm.write("Adding wild-type sequences to dataset...")

    # Validate name column exists
    if name_column not in dataset.columns:
        raise ValueError(f"Column '{name_column}' not found in dataset")

    # Create a copy to avoid modifying the original
    result_dataset = dataset.copy()
    result_dataset["error_message"] = None

    try:
        # Map sequences to dataset
        result_dataset["sequence"] = result_dataset[name_column].map(sequence_dict)

        # Mark missing sequences as errors
        missing_mask = result_dataset["sequence"].isnull()
        result_dataset.loc[missing_mask, "error_message"] = (
            "Sequence not found in sequence dictionary"
        )

        # Success mask is where we have sequences
        success_mask = ~missing_mask

        # Log missing proteins
        if missing_mask.any():
            missing_proteins = result_dataset[missing_mask][name_column].unique()
            tqdm.write(
                f"Warning: Missing sequences for {len(missing_proteins)} proteins: {list(missing_proteins[:10])}"
                + (" ..." if len(missing_proteins) > 10 else "")
            )

    except Exception as e:
        # If something goes wrong, mark all as failed
        result_dataset["error_message"] = f"Error mapping sequences: {str(e)}"
        success_mask = pd.Series([False] * len(result_dataset))

    # Separate successful and failed datasets
    successful_dataset = result_dataset[success_mask].drop(columns=["error_message"])
    failed_dataset = result_dataset[~success_mask].drop(columns=["sequence"])

    total_proteins = dataset[name_column].nunique()
    successful_proteins = successful_dataset[name_column].nunique()

    tqdm.write(
        f"Sequence addition: {len(successful_dataset)} successful, {len(failed_dataset)} failed "
        f"({successful_proteins}/{total_proteins} proteins)"
    )

    return successful_dataset, failed_dataset


@multiout_step(main="success", failed="failed")
def extract_domain_sequences(
    dataset: pd.DataFrame,
    sequence_column: str = "sequence",
    start_pos_column: str = "start_pos",
    end_pos_column: str = "end_pos",
    num_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract domain sequences from full sequences using position information

    This function extracts domain subsequences based on start and end positions.
    Records with invalid positions or missing sequences are separated into the failed dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset with full sequences and position information
    sequence_column : str, default='sequence'
        Column containing full wild-type sequences
    start_pos_column : str, default='start_pos'
        Column containing domain start positions (0-based)
    end_pos_column : str, default='end_pos'
        Column containing domain end positions
    num_workers : int, default=4
        Number of parallel workers for processing, set to -1 for all available CPUs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (successful_dataset, failed_dataset) - datasets with and without extraction errors

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'name': ['prot1', 'prot2', 'prot3'],
    ...     'sequence': ['ABCDEFGHIJ', 'KLMNOPQRST', None],
    ...     'start_pos': [2, 0, 5],
    ...     'end_pos': [7, 4, 10]
    ... })
    >>> successful, failed = extract_domain_sequences(df)
    >>> print(successful['sequence'].tolist())
    ['CDEFG', 'KLMN']
    >>> print(len(failed))  # Should be 1 (the None sequence)
    1
    """
    tqdm.write("Extracting domain sequences from full sequences...")

    # Validate required columns exist
    required_columns = [sequence_column, start_pos_column, end_pos_column]
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Create a copy to avoid modifying the original
    result_dataset = dataset.copy()

    # Prepare partial function with fixed parameters
    _extract_domain = partial(
        _extract_single_domain,
        dataset_columns=dataset.columns,
        sequence_column=sequence_column,
        start_pos_column=start_pos_column,
        end_pos_column=end_pos_column,
    )

    # Parallel processing
    rows = dataset.itertuples(index=False, name=None)
    results = Parallel(n_jobs=num_workers, backend="loky")(
        delayed(_extract_domain)(row)
        for row in tqdm(rows, total=len(dataset), desc="Extracting domains")
    )

    # Separate domain sequences and error messages
    domain_sequences, error_messages = map(list, zip(*results))

    result_dataset["domain_sequence"] = domain_sequences
    result_dataset["error_message"] = error_messages

    # Success mask is where we have domain sequences
    success_mask = pd.notnull(result_dataset["domain_sequence"])

    # For successful extractions, replace the original sequence with the domain sequence
    result_dataset.loc[success_mask, sequence_column] = result_dataset.loc[
        success_mask, "domain_sequence"
    ]

    # Separate successful and failed datasets
    successful_dataset = result_dataset[success_mask].drop(
        columns=["error_message", "domain_sequence", start_pos_column, end_pos_column]
    )
    failed_dataset = result_dataset[~success_mask].drop(columns=["domain_sequence"])

    tqdm.write(
        f"Domain extraction: {len(successful_dataset)} successful, {len(failed_dataset)} failed"
    )

    return successful_dataset, failed_dataset


def _extract_single_domain(
    row_data: Tuple,
    dataset_columns: pd.Index,
    sequence_column: str,
    start_pos_column: str,
    end_pos_column: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract domain sequence for a single row

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        (domain_sequence, error_message)
    """
    # Convert tuple to dict for easier access
    row = dict(zip(dataset_columns, row_data))

    try:
        full_seq = row.get(sequence_column)
        start_pos = row.get(start_pos_column)
        end_pos = row.get(end_pos_column)

        if pd.isnull(full_seq) or full_seq is None:
            return None, "Missing sequence"

        if pd.isnull(start_pos) or pd.isnull(end_pos):
            return None, "Missing position information"

        start_pos = int(start_pos)
        end_pos = int(end_pos)

        if start_pos < 0:
            return None, f"Invalid start position: {start_pos} < 0"

        if end_pos > len(full_seq):
            return (
                None,
                f"End position {end_pos} exceeds sequence length {len(full_seq)}",
            )

        if start_pos >= end_pos:
            return None, f"Invalid position range: start={start_pos} >= end={end_pos}"

        if start_pos >= len(full_seq):
            return (
                None,
                f"Start position {start_pos} exceeds sequence length {len(full_seq)}",
            )

        return full_seq[start_pos:end_pos], None

    except Exception as e:
        return None, f"Extraction error: {str(e)}"
