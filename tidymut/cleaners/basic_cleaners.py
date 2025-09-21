# tidymut/cleaners/basic_cleaners.py
from __future__ import annotations

import numpy as np
import pandas as pd
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm
from typing import cast, TYPE_CHECKING

from ..core.alphabet import ProteinAlphabet, DNAAlphabet, RNAAlphabet
from ..core.pipeline import pipeline_step, multiout_step
from ..core.sequence import ProteinSequence, DNASequence, RNASequence
from ..utils.cleaner_workers import (
    valid_single_mutation,
    apply_single_mutation,
    infer_wt_sequence_grouped,
    infer_single_mutationset,
)
from ..utils.dataset_builders import convert_format_1, convert_format_2
from ..utils.type_converter import (
    convert_data_types as _convert_data_types,
    convert_data_types_batch as _convert_data_types_batch,
)

if TYPE_CHECKING:
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
    )

__all__ = [
    "read_dataset",
    "merge_columns",
    "split_columns",
    "extract_and_rename_columns",
    "filter_and_clean_data",
    "convert_data_types",
    "validate_mutations",
    "apply_mutations_to_sequences",
    "infer_mutations_from_sequences",
    "infer_wildtype_sequences",
    "convert_to_mutation_dataset_format",
]


def __dir__() -> List[str]:
    return __all__


@pipeline_step
def read_dataset(
    file_path: Union[str, Path], file_format: Optional[str] = None, **kwargs
) -> pd.DataFrame:
    """
    Read dataset from specified file format and return as a pandas DataFrame.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the dataset file
    file_format : str
        Format of the dataset file ("csv", "tsv", "xlsx", etc.)
    kwargs : Dict[str, Any]
        Additional keyword arguments for file reading

    Returns
    -------
    pd.DataFrame
        Dataset loaded from the specified file

    Examples
    --------
    >>> # Specify file_format parameter
    >>> df = read_dataset("data.csv", "csv")
    >>>
    >>> # Detect file_format automatically
    >>> df = read_dataset("data.csv")
    """
    if file_format is None:
        file_format = Path(file_path).suffix.lstrip(".").lower()

    readers = {
        "csv": lambda path, **kw: pd.read_csv(path, **kw),
        "tsv": lambda path, **kw: pd.read_csv(path, sep="\t", **kw),
        "xlsx": lambda path, **kw: pd.read_excel(path, **kw),
    }

    tqdm.write(f"Reading dataset from {file_path}...")
    try:
        return readers[file_format](file_path, **kwargs)
    except KeyError:
        raise ValueError(f"Unsupported file format: {file_format}")


@pipeline_step
def merge_columns(
    dataset: pd.DataFrame,
    columns_to_merge: List[str],
    new_column_name: str,
    separator: str = "_",
    drop_original: bool = False,
    na_rep: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    custom_formatter: Optional[Callable[[pd.Series], str]] = None,
) -> pd.DataFrame:
    """Merge multiple columns into a single column using a separator

    This function combines values from multiple columns into a new column,
    with flexible formatting options.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset
    columns_to_merge : List[str]
        List of column names to merge
    new_column_name : str
        Name for the new merged column
    separator : str, default='_'
        Separator to use between values
    drop_original : bool, default=False
        Whether to drop the original columns after merging
    na_rep : Optional[str], default=None
        String representation of NaN values. If None, NaN values are skipped.
    prefix : Optional[str], default=None
        Prefix to add to the merged value
    suffix : Optional[str], default=None
        Suffix to add to the merged value
    custom_formatter : Optional[Callable], default=None
        Custom function to format each row. Takes a pd.Series and returns a string.
        If provided, ignores separator, prefix, suffix parameters.

    Returns
    -------
    pd.DataFrame
        Dataset with the new merged column

    Examples
    --------
    Basic usage:

    >>> df = pd.DataFrame({
    ...     'gene': ['BRCA1', 'TP53', 'EGFR'],
    ...     'position': [100, 200, 300],
    ...     'mutation': ['A', 'T', 'G']
    ... })
    >>> result = merge_columns(df, ['gene', 'position', 'mutation'], 'mutation_id', separator='_')
    >>> print(result['mutation_id'])
    0    BRCA1_100_A
    1     TP53_200_T
    2     EGFR_300_G

    With prefix and suffix:

    >>> result = merge_columns(
    ...     df, ['gene', 'position'], 'gene_pos',
    ...     separator=':', prefix='[', suffix=']'
    ... )
    >>> print(result['gene_pos'])
    0    [BRCA1:100]
    1     [TP53:200]
    2     [EGFR:300]

    Handling NaN values:

    >>> df_with_nan = pd.DataFrame({
    ...     'col1': ['A', 'B', None],
    ...     'col2': ['X', None, 'Z'],
    ...     'col3': [1, 2, 3]
    ... })
    >>> result = merge_columns(
    ...     df_with_nan, ['col1', 'col2', 'col3'], 'merged',
    ...     separator='-', na_rep='NA'
    ... )
    >>> print(result['merged'])
    0    A-X-1
    1    B-NA-2
    2    NA-Z-3

    Custom formatter:

    >>> def format_mutation(row):
    ...     return f"{row['gene']}:{row['position']}{row['mutation']}"
    >>> result = merge_columns(
    ...     df, ['gene', 'position', 'mutation'], 'hgvs',
    ...     custom_formatter=format_mutation
    ... )
    >>> print(result['hgvs'])
    0    BRCA1:100A
    1     TP53:200T
    2     EGFR:300G
    """
    tqdm.write(f"Merging columns {columns_to_merge} into '{new_column_name}'...")

    # Validate columns exist
    missing_cols = [col for col in columns_to_merge if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataset: {missing_cols}")

    # Create a copy to avoid modifying original
    result = dataset.copy()

    if custom_formatter is not None:
        # Use custom formatter
        tqdm.write("Using custom formatter...")
        tqdm.pandas()
        result[new_column_name] = result.progress_apply(custom_formatter, axis=1)  # type: ignore
    else:
        # Standard merging with separator
        df_to_merge = result[columns_to_merge].copy()

        if na_rep is not None:
            # Replace NaN with na_rep
            df_to_merge = df_to_merge.fillna(na_rep).astype(str)
        else:
            # Convert to string and replace NaN with empty string
            df_to_merge = df_to_merge.astype(str)
            mask = result[columns_to_merge].isna()
            df_to_merge = df_to_merge.mask(mask, "")

        # Vectorized merge
        merged = df_to_merge.agg(separator.join, axis=1)

        # Skip rows with all NaN values
        if na_rep is None:
            all_na = result[columns_to_merge].isna().all(axis=1)
            merged[all_na] = np.nan

        # Add prefix and suffix if specified
        if prefix is not None or suffix is not None:
            # Add prefix and suffix to non-NaN values
            non_na_mask = merged.notna()
            if prefix is not None:
                merged[non_na_mask] = prefix + merged[non_na_mask]
            if suffix is not None:
                merged[non_na_mask] = merged[non_na_mask] + suffix

        result[new_column_name] = merged

    # Drop original columns if requested
    if drop_original:
        result = result.drop(columns=columns_to_merge)
        tqdm.write(f"Dropped original columns: {columns_to_merge}")

    tqdm.write(f"Successfully created merged column '{new_column_name}'")
    return result


@pipeline_step
def split_columns(
    dataset: pd.DataFrame,
    column_to_split: str,
    new_column_names: List[str],
    separator: str = "_",
    max_splits: Optional[int] = None,
    drop_original: bool = False,
    fill_value: Optional[str] = "NaN",
    strip_whitespace: bool = True,
    regex: bool = False,
    custom_splitter: Optional[Callable[[str], List[str]]] = None,
) -> pd.DataFrame:
    """Split a single column into multiple columns using a separator

    This function splits values from one column into multiple new columns,
    with flexible splitting options.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset
    column_to_split : str
        Name of the column to split
    new_column_names : List[str]
        List of names for the new columns
    separator : str, default='_'
        Separator to use for splitting. Can be regex pattern if regex=True.
    max_splits : Optional[int], default=None
        Maximum number of splits to perform. If None, splits on all occurrences.
    drop_original : bool, default=False
        Whether to drop the original column after splitting
    fill_value : Optional[str], default=None
        Value to use when split results have fewer parts than new_column_names.
        If None, uses NaN for missing parts.
    strip_whitespace : bool, default=True
        Whether to strip whitespace from split results
    regex : bool, default=False
        Whether to treat separator as a regex pattern
    custom_splitter : Optional[Callable], default=None
        Custom function to split each value. Takes a string and returns a list of strings.
        If provided, ignores separator, max_splits, regex parameters.

    Returns
    -------
    pd.DataFrame
        Dataset with the new split columns

    Examples
    --------
    Basic usage:

    >>> df = pd.DataFrame({
    ...     'mutation_id': ['BRCA1_100_A', 'TP53_200_T', 'EGFR_300_G'],
    ...     'score': [0.95, 0.87, 0.92]
    ... })
    >>> result = split_columns(
    ...     df, 'mutation_id', ['gene', 'position', 'mutation'], separator='_'
    ... )
    Splitting column 'mutation_id' into ['gene', 'position', 'mutation']...
    Splitting using separator: '_'
    Successfully created split columns: ['gene', 'position', 'mutation']
    >>> print(result[['gene', 'position', 'mutation']])
        gene position mutation
    0  BRCA1      100        A
    1   TP53      200        T
    2   EGFR      300        G

    With max_splits:

    >>> df = pd.DataFrame({
    ...     'path': ['home/user/documents/file.txt', 'data/analysis/results.csv']
    ... })
    >>> result = split_columns(
    ...     df, 'path', ['root', 'rest'], separator='/', max_splits=1
    ... )
    Splitting column 'path' into ['root', 'rest']...
    Splitting using separator: '/'
    Successfully created split columns: ['root', 'rest']
    >>> print(result[['root', 'rest']])
       root                     rest
    0  home  user/documents/file.txt
    1  data     analysis/results.csv

    Handling insufficient splits with fill_value:

    >>> df = pd.DataFrame({
    ...     'incomplete': ['A_B', 'X_Y_Z', 'M']
    ... })
    >>> result = split_columns(
    ...     df, 'incomplete', ['col1', 'col2', 'col3'],
    ...     separator='_', fill_value='MISSING'
    ... )
    Splitting column 'incomplete' into ['col1', 'col2', 'col3']...
    Splitting using separator: '_'
    Successfully created split columns: ['col1', 'col2', 'col3']
    >>> print(result[['col1', 'col2', 'col3']])
      col1     col2     col3
    0    A        B  MISSING
    1    X        Y        Z
    2    M  MISSING  MISSING

    Using regex separator:

    >>> df = pd.DataFrame({
    ...     'text': ['word1-word2_word3', 'itemA|itemB-itemC']
    ... })
    >>> result = split_columns(
    ...     df, 'text', ['part1', 'part2', 'part3'],
    ...     separator=r'[-_|]', regex=True
    ... )
    Splitting column 'text' into ['part1', 'part2', 'part3']...
    Splitting using separator: '[-_|]' (regex)
    Successfully created split columns: ['part1', 'part2', 'part3']
    >>> print(result[['part1', 'part2', 'part3']])
       part1  part2  part3
    0  word1  word2  word3
    1  itemA  itemB  itemC

    Custom splitter:

    >>> def parse_coordinates(coord_str):
    ...     # Parse "chr1:12345-67890" format
    ...     parts = coord_str.replace(':', '_').replace('-', '_').split('_')
    ...     return parts
    >>> df = pd.DataFrame({
    ...     'coordinates': ['chr1:12345-67890', 'chr2:98765-43210']
    ... })
    >>> result = split_columns(
    ...     df, 'coordinates', ['chromosome', 'start', 'end'],
    ...     custom_splitter=parse_coordinates
    ... )
    Splitting column 'coordinates' into ['chromosome', 'start', 'end']...
    Using custom splitter...
    100%|█████████████████████████████| 2/2 [00:00<00:00, xx it/s]
    Successfully created split columns: ['chromosome', 'start', 'end']
    >>> print(result[['chromosome', 'start', 'end']])
      chromosome  start    end
    0       chr1  12345  67890
    1       chr2  98765  43210

    Handling NaN values:

    >>> df = pd.DataFrame({
    ...     'data': ['A_B_C', None, 'X_Y']
    ... })
    >>> result = split_columns(
    ...     df, 'data', ['col1', 'col2', 'col3'], separator='_'
    ... )
    Splitting column 'data' into ['col1', 'col2', 'col3']...
    Splitting using separator: '_'
    Successfully created split columns: ['col1', 'col2', 'col3']
    >>> print(result[['col1', 'col2', 'col3']])
       col1 col2 col3
    0     A    B    C
    1  None  NaN  NaN
    2     X    Y  NaN
    """
    tqdm.write(f"Splitting column '{column_to_split}' into {new_column_names}...")

    # Validate column exists
    if column_to_split not in dataset.columns:
        raise ValueError(f"Column '{column_to_split}' not found in dataset")

    # Validate new column names don't already exist
    existing_cols = [col for col in new_column_names if col in dataset.columns]
    if existing_cols:
        raise ValueError(f"New column names already exist in dataset: {existing_cols}")

    # Unify fill_value to NaN if not provided
    fill_value = fill_value if fill_value is not None else "NaN"
    # Create a copy to avoid modifying original
    result = dataset.copy()

    # Get the column to split
    col_series = result[column_to_split]

    if custom_splitter is not None:
        # Use custom splitter
        tqdm.write("Using custom splitter...")

        def apply_custom_splitter(value):
            if pd.isna(value):
                return [np.nan] * len(new_column_names)
            try:
                split_result = custom_splitter(str(value))
                # Pad or truncate to match new_column_names length
                padded_result = split_result[: len(new_column_names)]
                while len(padded_result) < len(new_column_names):
                    padded_result.append(fill_value)
                return padded_result
            except Exception as e:
                tqdm.write(f"Warning: Custom splitter failed for value '{value}': {e}")
                return ["NaN"] * len(new_column_names)

        tqdm.pandas()
        split_data = col_series.progress_apply(apply_custom_splitter)

        # Convert to DataFrame
        split_df = pd.DataFrame(
            split_data.tolist(), columns=new_column_names, index=result.index
        )

    else:
        # Standard splitting with separator
        tqdm.write(
            f"Splitting using separator: '{separator}'" + (" (regex)" if regex else "")
        )

        # Handle NaN values
        non_na_mask = col_series.notna()

        # Initialize result arrays
        split_df = pd.DataFrame(
            {
                col: pd.Series("NaN", index=result.index, dtype=object)
                for col in new_column_names
            }
        )

        if non_na_mask.any():
            # Convert to string and split
            str_series = col_series[non_na_mask].astype(str)
            temp_df = str_series.str.split(
                separator, n=(max_splits or -1), expand=True, regex=bool(regex)
            )
            if strip_whitespace:
                temp_df = temp_df.apply(lambda s: s.str.strip())

            # Ensure the resulting DataFrame has exactly the required number of columns
            # - If fewer, add extra columns filled with NaN
            # - If more, drop extra columns
            temp_df = temp_df.reindex(columns=range(len(new_column_names)))
            temp_df = temp_df.fillna(fill_value)

            # Assign each split column back into the result DataFrame
            # Alignment is preserved because we use .loc with the original index
            for i, col_name in enumerate(new_column_names):
                split_df.loc[str_series.index, col_name] = temp_df.iloc[:, i].astype(
                    object
                )

        # Post-process rows where the original value was missing:
        #   - first split column -> None;
        #   - remaining new columns -> fill_value
        missing_mask = ~non_na_mask
        if missing_mask.any():
            if new_column_names:
                # First column becomes None
                split_df.loc[missing_mask, new_column_names[0]] = None
                # Remaining columns become fill_value
                if len(new_column_names) > 1:
                    pad_value = fill_value
                    split_df.loc[missing_mask, new_column_names[1:]] = pad_value

    # Add new columns to result
    for col_name in new_column_names:
        result[col_name] = split_df[col_name]

    # Drop original column if requested
    if drop_original:
        result = result.drop(columns=[column_to_split])
        tqdm.write(f"Dropped original column: '{column_to_split}'")

    tqdm.write(f"Successfully created split columns: {new_column_names}")
    return result


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

    Examples
    --------
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

    Examples
    --------
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

    Examples
    --------
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
    exclude_patterns: Union[str, Sequence[str], Callable, None] = None,
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
    exclude : Union[str, List[str], callable, None], default=None
        Patterns to exclude from validation. Can be:
        - A regex pattern string (e.g., r'^WT$|^wildtype$')
        - A list of exact strings to match (e.g., ['WT', 'wildtype', 'UNKNOWN'])
        - A list containing both regex patterns (starting with 'regex:') and exact strings
        - A callable that takes a string value and returns True if it should be excluded
        - None to validate all values
    cache_results : bool, default=True
        Whether to cache formatting results for performance
    num_workers : int, default=4
        Number of parallel workers for processing, set to -1 for all available CPUs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (successful_dataset, failed_dataset) - datasets with valid and invalid mutations

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'name': ['protein1', 'protein1', 'protein2'],
    ...     'mut_info': ['A123S', 'C456D,E789F', 'InvalidMut'],
    ...     'score': [1.5, 2.3, 3.7]
    ... })
    >>> successful, failed = validate_mutations(df, mutation_column='mut_info', mutation_sep=',')
    >>> print(len(successful))
    2
    >>> print(successful['mut_info'].tolist())  # Formatted mutations
    ['A123S', 'C456D,E789F']
    >>> print(len(failed))
    1
    >>> print(failed['failed']['error_message'].iloc[0])  # Error message for failed mutation
    'ValueError: No valid mutations could be parsed...'
    >>> # Exclude patterns
    >>> df = pd.DataFrame({
    ...     'name': ['protein1', 'protein1', 'protein2', 'protein3', 'protein4'],
    ...     'mut_info': ['A123S', 'C456D,E789F', 'WT', 'wildtype', 'UNKNOWN'],
    ...     'score': [1.5, 2.3, 3.7, 4.1, 5.2]
    ... })
    >>>
    >>> # Exclude exact string matches
    >>> successful, failed = validate_mutations(
    ...     df,
    ...     mutation_column='mut_info',
    ...     exclude_patterns=['WT', 'wildtype', 'UNKNOWN']
    ... )
    >>>
    >>> # Exclude using regex pattern
    >>> successful, failed = validate_mutations(
    ...     df,
    ...     mutation_column='mut_info',
    ...     exclude_patterns=r'^(WT|wildtype|UNKNOWN)$'
    ... )
    >>>
    >>> # Mixed approach: regex patterns (prefixed with 'regex:') and exact strings
    >>> successful, failed = validate_mutations(
    ...     df,
    ...     mutation_column='mut_info',
    ...     exclude_patterns=['regex:^WT$', 'regex:.*type.*', 'UNKNOWN']
    ... )
    >>>
    >>> # Using a custom function
    >>> exclude_func = lambda x: str(x).upper() in ['WT', 'WILDTYPE'] or 'UNKNOWN' in str(x)
    >>> successful, failed = validate_mutations(
    ...     df,
    ...     mutation_column='mut_info',
    ...     exclude_patterns=exclude_func
    ... )
    >>> successful
           name     mut_info  score
    0  protein2           WT    3.7
    1  protein3     wildtype    4.1
    2  protein4      UNKNOWN    5.2
    3  protein1        A122S    1.5
    4  protein1  C455D,E788F    2.3
    """
    import re

    tqdm.write("Validating and formatting mutations...")

    if mutation_column not in dataset.columns:
        raise ValueError(f"Mutation column '{mutation_column}' not found")

    result = dataset.copy()
    original_len = len(result)

    # Prepare exclude function
    def should_exclude(value):
        if exclude_patterns is None:
            return False

        # Convert value to string for pattern matching
        str_value = str(value) if value is not None else ""

        if callable(exclude_patterns):
            # If exclude is a function, use it directly
            return exclude_patterns(value)
        elif isinstance(exclude_patterns, str):
            # If exclude is a single regex pattern
            try:
                return bool(re.search(exclude_patterns, str_value))
            except re.error:
                # If regex is invalid, treat as exact string match
                return str_value == exclude_patterns
        elif isinstance(exclude_patterns, (list, tuple)):
            # If exclude is a list of patterns/strings
            for pattern in exclude_patterns:
                if isinstance(pattern, str):
                    if pattern.startswith("regex:"):
                        # Handle explicit regex patterns
                        regex_pattern = pattern[6:]  # Remove 'regex:' prefix
                        try:
                            if re.search(regex_pattern, str_value):
                                return True
                        except re.error:
                            # If regex is invalid, skip this pattern
                            continue
                    else:
                        # Handle exact string match
                        if str_value == pattern:
                            return True
            return False
        else:
            # If exclude is neither callable nor string/list, treat as exact match
            return str_value == str(exclude_patterns)

    # Separate excluded and non-excluded rows
    mutation_values = result[mutation_column]
    exclude_mask = mutation_values.apply(should_exclude)
    # Rows that should be excluded from validation (treated as successful)
    excluded_dataset = result[exclude_mask].copy()
    # Rows that need validation
    validation_dataset = result[~exclude_mask].copy()

    if len(validation_dataset) == 0:
        # All rows were excluded, return all as successful
        tqdm.write(
            f"Mutation validation: {len(excluded_dataset)} excluded (treated as successful), "
            f"0 validated, 0 failed (out of {original_len} total)"
        )
        return excluded_dataset, pd.DataFrame(columns=result.columns)

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
    # Combine excluded rows with successful validated rows
    successful_dataset = pd.concat(
        [excluded_dataset, successful_dataset], ignore_index=True
    )

    # Create failed dataset
    failed_dataset = result_dataset[~success_mask].copy()
    failed_dataset = failed_dataset.drop(columns=["formatted_" + mutation_column])

    tqdm.write(
        f"Mutation validation: {len(excluded_dataset)} excluded (treated as successful), "
        f"{len(successful_dataset)} successful, {len(failed_dataset)} failed "
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
    is_zero_based: bool = True,
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
    is_zero_based : bool, default=True
        Whether origin mutation positions are zero-based
    sequence_type : str, default='protein'
        Type of sequence ('protein', 'dna', 'rna')
    num_workers : int, default=4
        Number of parallel workers for processing, set to -1 for all available CPUs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (successful_dataset, failed_dataset) - datasets with and without errors

    Examples
    --------
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
        is_zero_based=is_zero_based,
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


def infer_mutations_from_sequences(
    dataset: pd.DataFrame,
    wt_sequence_column: str = "wt_seq",
    mut_sequence_column: str = "mut_seq",
    mutation_sep: str = ",",
    sequence_type: str = "protein",
    num_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Infer mutations by comparing wild-type sequences with mutated sequences.

    This function takes wild-type and mutated sequences and identifies the
    mutations that occurred by comparing them position by position. It only
    handles sequences of equal length. Uses parallel processing for large datasets.

    Parameters
    ----------
    dataset : list of dict
        Input dataset containing wild-type and mutated sequence data.
        Each dict should contain the specified column keys.
    wt_sequence_column : str, default='wt_seq'
        Column key containing wild-type sequences
    mut_sequence_column : str, default='mut_seq'
        Column key containing mutated sequences
    mutation_sep : str, default=','
        Separator used to join multiple mutations in output string
    sequence_type : str, default='protein'
        Type of sequence ('protein', 'dna', 'rna')
    num_workers : int, default=4
        Number of parallel workers for processing, set to -1 for all available CPUs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (successful_results, failed_results) - datasets with and without errors

    Examples
    --------
    >>> dataset = pd.DataFrame([
    ...     {'name': 'prot1', 'wt_seq': 'AKCDEF', 'mut_seq': 'KKCDEF'},
    ...     {'name': 'prot1', 'wt_seq': 'AKCDEF', 'mut_seq': 'AKDDEF'},
    ...     {'name': 'prot2', 'wt_seq': 'FEGHIS', 'mut_seq': 'FFGHIS'},
    ...     {'name': 'prot3', 'wt_seq': 'ABC', 'mut_seq': 'ASC'}
    ... ])
    >>> successful, failed = infer_mutations_from_sequences(dataset)
    >>> successful
        name  wt_seq mut_seq inferred_mutations
    0  prot1  AKCDEF  KKCDEF                A0K
    1  prot1  AKCDEF  AKDDEF                C2D
    2  prot2  FEGHIS  FFGHIS                E1F
    >>> failed
        name wt_seq mut_seq                                  error_message
    3  prot3    ABC    ABCD  Invalid characters in Protein sequence: {'B'}
    """
    tqdm.write("Inferring mutations from sequence pairs...")

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

    # Create partial function for single row processing
    _infer_single_mutationset = partial(
        infer_single_mutationset,
        wt_sequence_column=wt_sequence_column,
        dataset_columns=dataset.columns,
        mut_sequence_column=mut_sequence_column,
        mutation_sep=mutation_sep,
        sequence_class=SequenceClass,
    )

    # Parallel processing
    rows = dataset.itertuples(index=False, name=None)
    results = Parallel(n_jobs=num_workers, backend="loky")(
        delayed(_infer_single_mutationset)(row)
        for row in tqdm(rows, desc="Processing sequences")
    )

    # Separate successful and failed results
    mutations, error_messages = map(list, zip(*results))

    result_dataset = dataset.copy()
    result_dataset["inferred_mutations"] = mutations
    result_dataset["error_message"] = error_messages

    success_mask = pd.notnull(result_dataset["inferred_mutations"])
    successful_dataset = result_dataset[success_mask].drop(columns=["error_message"])
    failed_dataset = result_dataset[~success_mask].drop(columns=["inferred_mutations"])

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
        Number of parallel workers for processing, set to -1 for all available CPUs

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (successful_dataset, problematic_dataset) - datasets with added WT rows

    Examples
    --------
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
    label_column: str = "score",
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

    label_column : str, default='score'
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

    Format 1: With WT rows and multi-mutations

    >>> df1 = pd.DataFrame({
    ...     'name': ['prot1', 'prot1', 'prot1', 'prot2', 'prot2'],
    ...     'mut_info': ['A0S,Q1D', 'C2D', 'WT', 'E0F', 'WT'],
    ...     'mut_seq': ['SDCDEF', 'AQDDEF', 'AQCDEF', 'FGHIGHK', 'EGHIGHK'],
    ...     'score': [1.5, 2.0, 0.0, 3.0, 0.0]
    ... })
    >>> result_df1, ref_seqs1 = convert_to_mutation_dataset_format(df1)
    >>> # Input has 5 rows but output has 6 rows (A0S,Q1D -> 2 rows)

    Format 2: With sequence column and multi-mutations

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
    basic_required = [name_column, mutation_column, label_column]
    if mutated_sequence_column:
        basic_required.append(mutated_sequence_column)

    missing_basic = [col for col in basic_required if col not in df.columns]
    if missing_basic:
        raise ValueError(f"Missing required columns: {missing_basic}")

    # Convert `mutation_column`` to uppercase for consistency
    df[mutation_column] = df[mutation_column].str.upper()

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
    has_wt_rows = (
        mutation_column in df.columns and df[mutation_column].str.contains("WT").any()
    )

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
            label_column,
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
            label_column,
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
                label_column,
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
                label_column,
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
