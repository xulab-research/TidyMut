# tidymut/cleaners/basic_cleaners.py

import numpy as np
import pandas as pd
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from ..core.alphabet import ProteinAlphabet, DNAAlphabet, RNAAlphabet
from ..core.pipeline import pipeline_step, multiout_step
from ..core.sequence import ProteinSequence, DNASequence, RNASequence
from ..utils.cleaner_workers import (
    valid_single_mutation,
    apply_single_mutation,
    infer_wt_sequence_grouped,
)
from ..utils.dataset_builders import convert_format_1, convert_format_2
from ..utils.type_converter import (
    convert_data_types as _convert_data_types,
    convert_data_types_batch as _convert_data_types_batch,
)


@pipeline_step
def extract_and_rename_columns(
    dataset: pd.DataFrame,
    column_mapping: Dict[str, str],
    required_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Extract useful columns and rename them to standard format.

    This function extracts specified columns from the input dataset and renames them
    according to the provided mapping. It helps standardize column names across
    different datasets.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing the data to be processed
    column_mapping : Dict[str, str]
        Column name mapping from original names to new names
        Format: {original_column_name: new_column_name}
    required_columns : Optional[Sequence[str]], default=None
        Required column names. If None, extracts all mapped columns

    Returns
    -------
    pd.DataFrame
        Dataset with extracted and renamed columns

    Raises
    ------
    ValueError
        If required columns are missing from the input dataset

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'uniprot_ID': ['P12345', 'Q67890'],
    ...     'mutation_type': ['A123B', 'C456D'],
    ...     'score_value': [1.5, -2.3],
    ...     'extra_col': ['x', 'y']
    ... })
    >>> mapping = {
    ...     'uniprot_ID': 'name',
    ...     'mutation_type': 'mut_info',
    ...     'score_value': 'label'
    ... }
    >>> result = extract_and_rename_columns(df, mapping)
    >>> print(result.columns.tolist())
    ['name', 'mut_info', 'label']
    """
    tqdm.write("Extracting and renaming columns...")

    # Check if required columns exist
    missing_cols = [col for col in column_mapping.keys() if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract and rename columns
    if required_columns:
        # Only extract specified columns
        extract_cols = [
            col
            for col in column_mapping.keys()
            if column_mapping[col] in required_columns
        ]
        extracted_dataset = dataset[extract_cols].copy()
    else:
        # Extract all mapped columns
        extracted_dataset = dataset[list(column_mapping.keys())].copy()

    # Rename columns
    extracted_dataset = extracted_dataset.rename(columns=column_mapping)

    tqdm.write(
        f"Extracted {len(extracted_dataset.columns)} columns: {list(extracted_dataset.columns)}"
    )
    return extracted_dataset


@pipeline_step
def filter_and_clean_data(
    dataset: pd.DataFrame,
    filters: Optional[Dict[str, Union[Any, Callable[[pd.Series], pd.Series]]]] = None,
    exclude_patterns: Optional[Dict[str, Union[str, List[str]]]] = None,
    drop_na_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter and clean data based on specified conditions.

    This function provides flexible data filtering and cleaning capabilities,
    including value-based filtering, pattern exclusion, and null value removal.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset to be filtered and cleaned
    filters : Optional[Dict[str, Union[Any, Callable[[pd.Series], pd.Series]]]], default=None
        Filter conditions in format {column_name: condition_value_or_function}
        If value is callable, it will be applied to the column
    exclude_patterns : Optional[Dict[str, Union[str, List[str]]]], default=None
        Exclusion patterns in format {column_name: regex_pattern_or_list}
        Rows matching these patterns will be excluded
    drop_na_columns : Optional[List[str]], default=None
        List of column names where null values should be dropped

    Returns
    -------
    pd.DataFrame
        Filtered and cleaned dataset

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'mut_type': ['A123B', 'wt', 'C456D', 'insert', 'E789F'],
    ...     'score': [1.5, 2.0, '-', 3.2, 4.1],
    ...     'quality': ['good', 'bad', 'good', 'good', None]
    ... })
    >>> filters = {'score': lambda x: x != '-'}
    >>> exclude_patterns = {'mut_type': ['wt', 'insert']}
    >>> drop_na_columns = ['quality']
    >>> result = filter_and_clean_data(df, filters, exclude_patterns, drop_na_columns)
    >>> print(len(result))  # Should be 2 (A123B and E789F rows)
    2
    """
    tqdm.write("Filtering and cleaning data...")
    original_len = len(dataset)

    # Collect all filter conditions to avoid dataframe copy
    filter_masks = []

    available_columns = set(dataset.columns)

    # Apply filter conditions
    if filters:
        for col, condition in filters.items():
            if col not in available_columns:
                tqdm.write(f"Warning: Column '{col}' not found for filtering")
                continue

            if callable(condition):
                mask = condition(dataset[col])
                filter_masks.append(mask)
            else:
                mask = dataset[col] == condition
                filter_masks.append(mask)

    # Exclude specific patterns
    if exclude_patterns:
        for col, patterns in exclude_patterns.items():
            if col not in available_columns:
                tqdm.write(f"Warning: Column '{col}' not found for pattern exclusion")
                continue

            if isinstance(patterns, str):
                patterns = [patterns]

            # Combine patterns into a single regex pattern
            if len(patterns) == 1:
                combined_pattern = patterns[0]
            else:
                combined_pattern = "|".join(f"({pattern})" for pattern in patterns)

            mask = ~dataset[col].str.contains(combined_pattern, na=False, regex=True)
            filter_masks.append(mask)

    # Drop null values for specified columns
    if drop_na_columns:
        for col in drop_na_columns:
            if col in available_columns:
                mask = dataset[col].notna()
                filter_masks.append(mask)

    # Apply combined filter conditions
    if filter_masks:
        combined_mask = filter_masks[0]
        for mask in filter_masks[1:]:
            combined_mask &= mask

        result = dataset.loc[combined_mask].copy()
    else:
        result = dataset.copy()

    tqdm.write(
        f"Filtered data: {original_len} -> {len(result)} rows "
        f"({len(result)/original_len*100:.1f}% retained)"
    )
    return result


@pipeline_step
def convert_data_types(
    dataset: pd.DataFrame,
    type_conversions: Dict[str, Union[str, Type, np.dtype]],
    handle_errors: str = "coerce",
    optimize_memory: bool = True,
    use_batch_processing: bool = False,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """
    Convert data types for specified columns.

    This function provides unified data type conversion with error handling options.
    Supports pandas, numpy, and Python built-in types with memory optimization.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset with columns to be converted
    type_conversions : Dict[str, Union[str, Type, np.dtype]]
        Type conversion mapping in format {column_name: target_type}
        Supported formats:
        - String types: 'float', 'int', 'str', 'category', 'bool', 'datetime'
        - Numpy types: np.float32, np.float64, np.int32, np.int64, etc.
        - Pandas types: 'Int64', 'Float64', 'string', 'boolean'
        - Python types: float, int, str, bool
    handle_errors : str, default='coerce'
        Error handling strategy: 'raise', 'coerce', or 'ignore'
    optimize_memory : bool, default=True
        Whether to automatically optimize memory usage by choosing smaller dtypes
    use_batch_processing : bool, default=False
        Whether to use batch processing for large datasets
    chunk_size : int, default=10000
        Chunk size when using batch processing

    Returns
    -------
    pd.DataFrame
        Dataset with converted data types

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'score': ['1.5', '2.3', '3.7'],
    ...     'count': ['10', '20', '30'],
    ...     'name': [123, 456, 789],
    ...     'flag': ['True', 'False', 'True']
    ... })
    >>> conversions = {
    ...     'score': np.float32,
    ...     'count': 'Int64',
    ...     'name': 'string',
    ...     'flag': 'boolean'
    ... }
    >>> result = convert_data_types(df, conversions)
    """
    tqdm.write("Converting data types...")

    if use_batch_processing:
        return _convert_data_types_batch(
            dataset, type_conversions, handle_errors, optimize_memory, chunk_size
        )
    else:
        return _convert_data_types(
            dataset, type_conversions, handle_errors, optimize_memory
        )


@multiout_step(main="success", failed="failed")
def validate_mutations(
    dataset: pd.DataFrame,
    mutation_column: str = "mut_info",
    format_mutations: bool = True,
    mutation_sep: str = ",",
    is_zero_based: bool = False,
    cache_results: bool = True,
    num_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and format mutation information.

    This function validates mutation strings, optionally formats them to a standard
    representation, and separates valid and invalid mutations into different datasets.
    It supports caching for improved performance on datasets with repeated mutations.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing mutation information
    mutation_column : str, default='mut_info'
        Name of the column containing mutation information
    format_mutations : bool, default=True
        Whether to format mutations to standard representation
    mutation_sep : str, default=','
        Separator used to split multiple mutations in a single string (e.g., 'A123B,C456D')
    is_zero_based : bool, default=False
        Whether origin mutation positions are zero-based
    cache_results : bool, default=True
        Whether to cache formatting results for performance
    num_workers : int, default=4
        Number of parallel workers for processing

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (successful_dataset, failed_dataset) - datasets with valid and invalid mutations

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'name': ['protein1', 'protein1', 'protein2'],
    ...     'mut_info': ['A123S', 'C456D,E789F', 'InvalidMut'],
    ...     'score': [1.5, 2.3, 3.7]
    ... })
    >>> successful, failed = validate_mutations(df, mutation_column='mut_info', mutation_sep=',')
    >>> print(len(successful))  # Should be 2 (valid mutations)
    2
    >>> print(successful['mut_info'].tolist())  # Formatted mutations
    ['A123S', 'C456D,E789F']
    >>> print(len(failed))  # Should be 1 (invalid mutation)
    1
    >>> print(failed['failed']['error_message'].iloc[0])  # Error message for failed mutation
    'ValueError: No valid mutations could be parsed...'
    """
    tqdm.write("Validating and formatting mutations...")

    if mutation_column not in dataset.columns:
        raise ValueError(f"Mutation column '{mutation_column}' not found")

    result = dataset.copy()
    original_len = len(result)

    # Global cache for parallel processing (shared memory)
    if cache_results:
        from multiprocessing import Manager

        manager = Manager()
        cache = manager.dict()
    else:
        cache = None

    # Prepare arguments for parallel processing
    mutation_values = result[mutation_column].tolist()
    args_list = [
        (
            mut_info,
            format_mutations,
            mutation_sep,
            is_zero_based,
            cache if cache_results else None,
        )
        for mut_info in mutation_values
    ]

    # Parallel processing
    results = Parallel(n_jobs=num_workers, backend="loky")(
        delayed(valid_single_mutation)(args)
        for args in tqdm(args_list, desc="Processing mutations")
    )

    # Separate formatted mutations and error messages
    formatted_mutations, error_messages = map(list, zip(*results))

    # Add results to dataset
    result_dataset = result.copy()
    result_dataset["formatted_" + mutation_column] = formatted_mutations
    result_dataset["error_message"] = error_messages

    # Create success mask based on whether formatted mutation is available
    success_mask = pd.notnull(result_dataset["formatted_" + mutation_column])

    # Create successful dataset
    successful_dataset = result_dataset[success_mask].copy()
    if format_mutations:
        # Replace original mutation column with formatted version
        successful_dataset[mutation_column] = successful_dataset[
            "formatted_" + mutation_column
        ]
    successful_dataset = successful_dataset.drop(
        columns=["formatted_" + mutation_column, "error_message"]
    )

    # Create failed dataset
    failed_dataset = result_dataset[~success_mask].copy()
    failed_dataset = failed_dataset.drop(columns=["formatted_" + mutation_column])

    tqdm.write(
        f"Mutation validation: {len(successful_dataset)} successful, {len(failed_dataset)} failed "
        f"(out of {original_len} total, {len(successful_dataset)/original_len*100:.1f}% valid)"
    )

    return successful_dataset, failed_dataset


@multiout_step(main="success", failed="failed")
def apply_mutations_to_sequences(
    dataset: pd.DataFrame,
    sequence_column: str = "sequence",
    name_column: str = "name",
    mutation_column: str = "mut_info",
    position_columns: Optional[Dict[str, str]] = None,
    mutation_sep: str = ",",
    sequence_type: str = "protein",
    num_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply mutations to sequences to generate mutated sequences.

    This function takes mutation information and applies it to wild-type sequences
    to generate the corresponding mutated sequences. It supports parallel processing
    and can handle position-based sequence extraction.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing mutation information and sequence data
    sequence_column : str, default='sequence'
        Column name containing wild-type sequences
    name_column : str, default='name'
        Column name containing protein identifiers
    mutation_column : str, default='mut_info'
        Column name containing mutation information
    position_columns : Optional[Dict[str, str]], default=None
        Position column mapping {"start": "start_col", "end": "end_col"}
        Used for extracting sequence regions
    mutation_sep : str, default=','
        Separator used to split multiple mutations in a single string
    sequence_type : str, default='protein'
        Type of sequence ('protein', 'dna', 'rna')
    num_workers : int, default=4
        Number of parallel workers for processing

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (successful_dataset, failed_dataset) - datasets with and without errors

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'name': ['prot1', 'prot1', 'prot2'],
    ...     'sequence': ['AKCDEF', 'AKCDEF', 'FEGHIS'],
    ...     'mut_info': ['A0K', 'C2D', 'E1F'],
    ...     'score': [1.0, 2.0, 3.0]
    ... })
    >>> successful, failed = apply_mutations_to_sequences(df)
    >>> print(successful['mut_seq'].tolist())
    ['KKCDEF', 'AKDDEF', 'FFGHIS']
    >>> print(len(failed))  # Should be 0 if all mutations are valid
    0
    """
    tqdm.write("Applying mutations to sequences...")

    # Validate required columns exist
    required_columns = [sequence_column, name_column, mutation_column]
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Select appropriate sequence class based on sequence_type
    sequence_type = sequence_type.lower()
    if sequence_type == "protein":
        SequenceClass = ProteinSequence
    elif sequence_type == "dna":
        SequenceClass = DNASequence
    elif sequence_type == "rna":
        SequenceClass = RNASequence
    else:
        raise ValueError(
            f"Unsupported sequence type: {sequence_type}. Must be 'protein', 'dna', or 'rna'"
        )

    _apply_single_mutation = partial(
        apply_single_mutation,
        dataset_columns=dataset.columns,
        sequence_column=sequence_column,
        name_column=name_column,
        mutation_column=mutation_column,
        position_columns=position_columns,
        mutation_sep=mutation_sep,
        sequence_class=SequenceClass,
    )

    # Parallel processing
    rows = dataset.itertuples(index=False, name=None)
    results = Parallel(n_jobs=num_workers, backend="loky")(
        delayed(_apply_single_mutation)(row)
        for row in tqdm(rows, total=len(dataset), desc="Applying mutations")
    )

    # Separate successful and failed results
    mutated_seqs, error_messages = map(list, zip(*results))

    result_dataset = dataset.copy()
    result_dataset["mut_seq"] = mutated_seqs
    result_dataset["error_message"] = error_messages

    success_mask = pd.notnull(result_dataset["mut_seq"])
    successful_dataset = result_dataset[success_mask].drop(columns=["error_message"])
    failed_dataset = result_dataset[~success_mask].drop(columns=["mut_seq"])

    tqdm.write(
        f"Mutation application: {len(successful_dataset)} successful, {len(failed_dataset)} failed"
    )
    return successful_dataset, failed_dataset


@multiout_step(main="successful", failed="failed")
def infer_wildtype_sequences(
    dataset: pd.DataFrame,
    name_column: str = "name",
    mutation_column: str = "mut_info",
    sequence_column: str = "mut_seq",
    label_columns: Optional[List[str]] = None,
    wt_label: float = 0.0,
    mutation_sep: str = ",",
    is_zero_based: bool = False,
    sequence_type: Literal["protein", "dna", "rna"] = "protein",
    handle_multiple_wt: Literal["error", "separate", "first"] = "error",
    num_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Infer wild-type sequences from mutated sequences and add WT rows.

    This function takes mutated sequences and their corresponding mutations to
    infer the original wild-type sequences. For each protein, it adds WT row(s)
    to the dataset with the inferred wild-type sequence.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing mutated sequences and mutation information
    name_column : str, default='name'
        Column name containing protein identifiers
    mutation_column : str, default='mut_info'
        Column name containing mutation information
    sequence_column : str, default='mut_seq'
        Column name containing mutated sequences
    label_columns : Optional[List[str]], default=None
        List of label column names to preserve
    wt_label : float, default=0.0
        Wild type score for WT rows
    mutation_sep : str, default=','
        Separator used to split multiple mutations in a single string
    is_zero_based : bool, default=False
        Whether origin mutation positions are zero-based
    sequence_type : str, default='protein'
        Type of sequence ('protein', 'dna', 'rna')
    handle_multiple_wt : Literal["error", "separate", "first"], default='error'
        How to handle multiple wild-type sequences: 'separate', 'first', or 'error'
    num_workers : int, default=4
        Number of parallel workers for processing

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (successful_dataset, problematic_dataset) - datasets with added WT rows

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'name': ['prot1', 'prot1', 'prot2'],
    ...     'mut_info': ['A0S', 'C2D', 'E0F'],
    ...     'mut_seq': ['SQCDEF', 'AQDDEF', 'FGHIGHK'],
    ...     'score': [1.0, 2.0, 3.0]
    ... })
    >>> success, failed = infer_wildtype_sequences(
    ...     df, label_columns=['score']
    ... )
    >>> print(len(success))  # Should have original rows + WT rows
    """
    tqdm.write("Inferring wildtype sequences...")

    if label_columns is None:
        label_columns = [col for col in dataset.columns if col.startswith("label_")]

    # Select appropriate sequence class based on sequence_type
    if sequence_type.lower() == "protein":
        SequenceClass = ProteinSequence
        AlphabetClass = ProteinAlphabet
    elif sequence_type.lower() == "dna":
        SequenceClass = DNASequence
        AlphabetClass = DNAAlphabet
    elif sequence_type.lower() == "rna":
        SequenceClass = RNASequence
        AlphabetClass = RNAAlphabet
    else:
        raise ValueError(
            f"Unsupported sequence type: {sequence_type.lower()}. Must be 'protein', 'dna', or 'rna'"
        )

    _process_protein_group = partial(
        infer_wt_sequence_grouped,
        name_column=name_column,
        mutation_column=mutation_column,
        sequence_column=sequence_column,
        label_columns=label_columns,
        wt_label=wt_label,
        mutation_sep=mutation_sep,
        is_zero_based=is_zero_based,
        handle_multiple_wt=handle_multiple_wt,
        sequence_class=SequenceClass,
        alphabet_class=AlphabetClass,
    )

    # Group by protein and process in parallel
    grouped = list(dataset.groupby(name_column, sort=False))

    try:
        results = Parallel(n_jobs=num_workers, backend="loky")(
            delayed(_process_protein_group)(group_data)
            for group_data in tqdm(grouped, desc="Processing proteins")
        )
    except Exception as e:
        tqdm.write(
            f"Warning: Parallel processing failed, falling back to sequential: {e}"
        )
        # Fallback to sequential processing
        results = []
        for group_data in tqdm(grouped, desc="Processing proteins (sequential)"):
            try:
                result = _process_protein_group(group_data)
                results.append(result)
            except Exception as group_e:
                # Create error entry for this specific group
                protein_name = group_data[0]
                error_row = {
                    name_column: str(protein_name),
                    "error_message": f"Sequential processing error: {type(group_e).__name__}: {str(group_e)}",
                }
                results.append(([error_row], "failed"))

    # Filter out None results and validate structure
    valid_results = []
    invalid_count = 0

    for i, result in enumerate(results):
        if result is None:
            invalid_count += 1
            tqdm.write(f"Warning: Result {i} is None, skipping")
            continue

        if not isinstance(result, tuple) or len(result) != 2:
            invalid_count += 1
            tqdm.write(f"Warning: Result {i} has invalid format, skipping: {result}")
            continue

        rows_list, category = result
        if category not in ("success", "failed"):
            invalid_count += 1
            tqdm.write(
                f"Warning: Result {i} has invalid category '{category}', skipping"
            )
            continue

        if not isinstance(rows_list, list):
            invalid_count += 1
            tqdm.write(f"Warning: Result {i} has invalid rows format, skipping")
            continue

        valid_results.append(result)

    if invalid_count > 0:
        tqdm.write(f"Warning: {invalid_count} invalid results were skipped")

    # Collect all rows
    successful_rows = []
    failed_rows = []

    for rows_list, category in valid_results:
        if category == "success":
            successful_rows.extend(rows_list)
        else:
            failed_rows.extend(rows_list)

    # Convert to DataFrame format
    successful_df = pd.DataFrame(successful_rows) if successful_rows else pd.DataFrame()
    failed_df = pd.DataFrame(failed_rows) if failed_rows else pd.DataFrame()

    tqdm.write(
        f"Wildtype inference: {len(successful_rows)} successful rows, {len(failed_rows)} failed rows"
    )
    tqdm.write(
        f"Added WT rows for proteins. Success: {len(successful_df)}, Failed: {len(failed_df)}"
    )

    return successful_df, failed_df


@pipeline_step
def convert_to_mutation_dataset_format(
    df: pd.DataFrame,
    name_column: str = "name",
    mutation_column: str = "mut_info",
    sequence_column: Optional[str] = None,
    mutated_sequence_column: str = "mut_seq",
    sequence_type: Literal["protein", "dna", "rna"] = "protein",
    score_column: str = "score",
    include_wild_type: bool = False,
    mutation_set_prefix: str = "set",
    is_zero_based: bool = False,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Convert a mutation DataFrame to the format required by MutationDataset.from_dataframe().

    This function supports two input formats:
    1. Format with WT rows: Contains explicit 'WT' entries with wild-type sequences
    2. Format with sequence column: Each row contains the wild-type sequence

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Supports two formats:

        Format 1 (with WT rows):
        - name: protein identifier
        - mut_info: mutation info ('A0S') or 'WT' for wild-type
        - mut_seq: mutated or wild-type sequence
        - score: numerical score

        Format 2 (with sequence column):
        - name: protein identifier
        - sequence: wild-type sequence
        - mut_info: mutation info ('A0S')
        - mut_seq: mutated sequence
        - score: numerical score

    name_column : str, default='name'
        Column name containing protein/sequence identifiers.

    mutation_column : str, default='mut_info'
        Column name containing mutation information. Expected formats:
        - 'A0S': amino acid mutation (wild_type + position + mutant_type)
        - 'WT': wild-type sequence (only in Format 1)

    sequence_column : Optional[str], default=None
        Column name containing wild-type sequences (Format 2 only).
        If provided, assumes Format 2. If None, assumes Format 1.

    mutated_sequence_column : Optional[str], default='mut_seq'
        Column name containing the mutated sequences.

    score_column : str, default='score'
        Column name containing scores or other numerical values.

    include_wild_type : bool, default=False
        Whether to include wild-type (WT) entries in the output. Only applies
        to Format 1 with explicit WT rows.

    mutation_set_prefix : str, default='set'
        Prefix used for generating mutation set IDs.

    is_zero_based : bool, default=False
        Whether mutation positions are zero-based.

    additional_metadata : Optional[Dict[str, Any]], default=None
        Additional metadata to add to all mutation sets.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, str]]
        (converted_dataframe, reference_sequences_dict)

        converted_dataframe: DataFrame in MutationDataset.from_dataframe() format
        reference_sequences_dict: Dictionary mapping reference_id to wild-type sequences
        (extracted from WT rows in Format 1 or sequence column in Format 2)

    Raises
    ------
    ValueError
        If required columns are missing or mutation strings cannot be parsed.

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Format 1: With WT rows and multi-mutations
    >>> df1 = pd.DataFrame({
    ...     'name': ['prot1', 'prot1', 'prot1', 'prot2', 'prot2'],
    ...     'mut_info': ['A0S,Q1D', 'C2D', 'WT', 'E0F', 'WT'],
    ...     'mut_seq': ['SDCDEF', 'AQDDEF', 'AQCDEF', 'FGHIGHK', 'EGHIGHK'],
    ...     'score': [1.5, 2.0, 0.0, 3.0, 0.0]
    ... })
    >>> result_df1, ref_seqs1 = convert_to_mutation_dataset_format(df1)
    >>> # Input has 5 rows but output has 6 rows (A0S,Q1D -> 2 rows)
    >>>
    >>> # Format 2: With sequence column and multi-mutations
    >>> df2 = pd.DataFrame({
    ...     'name': ['prot1', 'prot1', 'prot2'],
    ...     'sequence': ['AKCDEF', 'AKCDEF', 'FEGHIS'],
    ...     'mut_info': ['A0K,C2D', 'Q1P', 'E1F'],
    ...     'score': [1.5, 2.0, 3.0],
    ...     'mut_seq': ['KKDDEF', 'APCDEF', 'FFGHIS']
    ... })
    >>> result_df2, ref_seqs2 = convert_to_mutation_dataset_format(
    ...     df2, sequence_column='sequence'
    ... )
    >>> print(ref_seqs2['prot1'])
    AKCDEF
    >>> # First row generates 2 output rows for A0K and C2D mutations
    """
    # Validate input DataFrame
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")

    # Check basic required columns first
    basic_required = [name_column, mutation_column, score_column]
    if mutated_sequence_column:
        basic_required.append(mutated_sequence_column)

    missing_basic = [col for col in basic_required if col not in df.columns]
    if missing_basic:
        raise ValueError(f"Missing required columns: {missing_basic}")

    # Select appropriate sequence class based on sequence_type
    if sequence_type.lower() == "protein":
        SequenceClass = ProteinSequence
    elif sequence_type.lower() == "dna":
        SequenceClass = DNASequence
    elif sequence_type.lower() == "rna":
        SequenceClass = RNASequence
    else:
        raise ValueError(
            f"Unsupported sequence type: {sequence_type.lower()}. Must be 'protein', 'dna', or 'rna'"
        )

    # Intelligently determine input format based on actual data content
    has_sequence_column = sequence_column is not None and sequence_column in df.columns
    has_wt_rows = mutation_column in df.columns and "WT" in df[mutation_column].values

    # Decision logic for format detection
    if has_sequence_column and not has_wt_rows:
        # Clearly Format 2: has sequence column, no WT rows
        tqdm.write(
            f"Detected Format 2: Found sequence column '{sequence_column}', no WT rows"
        )
        sequence_column = cast(str, sequence_column)
        return convert_format_2(
            df,
            name_column,
            mutation_column,
            sequence_column,
            score_column,
            mutation_set_prefix,
            is_zero_based,
            additional_metadata,
            SequenceClass,
        )
    elif has_wt_rows and not has_sequence_column:
        # Clearly Format 1: has WT rows, no sequence column
        tqdm.write(f"Detected Format 1: Found WT rows, no sequence column")
        return convert_format_1(
            df,
            name_column,
            mutation_column,
            mutated_sequence_column,
            score_column,
            include_wild_type,
            mutation_set_prefix,
            is_zero_based,
            additional_metadata,
            SequenceClass,
        )
    elif has_sequence_column and has_wt_rows:
        # Ambiguous: has both sequence column and WT rows
        # Prefer Format 2 if sequence column was explicitly specified
        if sequence_column is not None:
            tqdm.write(
                f"Warning: Found both sequence column '{sequence_column}' and WT rows. "
                f"Using Format 2 as sequence_column was specified."
            )
            return convert_format_2(
                df,
                name_column,
                mutation_column,
                sequence_column,
                score_column,
                mutation_set_prefix,
                is_zero_based,
                additional_metadata,
                SequenceClass,
            )
        else:
            tqdm.write(
                "Warning: Found WT rows but sequence column exists. Using Format 1."
            )
            return convert_format_1(
                df,
                name_column,
                mutation_column,
                mutated_sequence_column,
                score_column,
                include_wild_type,
                mutation_set_prefix,
                is_zero_based,
                additional_metadata,
                SequenceClass,
            )
    else:
        # Neither format detected
        error_msg = "Cannot determine input format:\n"
        if sequence_column is not None:
            error_msg += f"  - Sequence column '{sequence_column}' specified but not found in DataFrame\n"
        error_msg += f"  - No 'WT' entries found in '{mutation_column}' column\n"
        error_msg += (
            "Please ensure your DataFrame matches one of the supported formats:\n"
        )
        error_msg += "  Format 1: Include 'WT' rows with wild-type sequences\n"
        error_msg += "  Format 2: Include a sequence column with wild-type sequences"
        raise ValueError(error_msg)
