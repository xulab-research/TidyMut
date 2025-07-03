# tidymut/cleaners/basic_cleaners.py
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import requests
import time
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm
from typing import cast, TYPE_CHECKING
from urllib.parse import urlparse

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
    "download_origin_data",
    "read_dataset",
    "merge_columns",
    "extract_and_rename_columns",
    "filter_and_clean_data",
    "convert_data_types",
    "validate_mutations",
    "apply_mutations_to_sequences",
    "infer_wildtype_sequences",
    "convert_to_mutation_dataset_format",
]


def __dir__() -> List[str]:
    return __all__


@pipeline_step
def download_origin_data(
    url: str,
    local_path: Union[str, Path],
    overwrite: bool = False,
    chunk_size: int = 8192,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    headers: Optional[Dict[str, str]] = None,
    verify_ssl: bool = True,
    expected_checksum: Optional[str] = None,
    checksum_algorithm: str = "md5",
    show_progress: bool = True,
    create_dirs: bool = True,
) -> Path:
    """
    Download data from a URL and save to local path with progress tracking.

    Parameters
    ----------
    url : str
        URL to download data from
    local_path : Union[str, Path]
        Local path where the downloaded file will be saved
    overwrite : bool, default=False
        Whether to overwrite existing files
    chunk_size : int, default=8192
        Size of chunks to download at a time (in bytes)
    timeout : int, default=30
        Request timeout in seconds
    max_retries : int, default=3
        Maximum number of retry attempts if download fails
    retry_delay : float, default=1.0
        Delay between retry attempts in seconds
    headers : Optional[Dict[str, str]], default=None
        Additional HTTP headers to send with the request
    verify_ssl : bool, default=True
        Whether to verify SSL certificates
    expected_checksum : Optional[str], default=None
        Expected checksum of the downloaded file for verification
    checksum_algorithm : str, default="md5"
        Algorithm to use for checksum verification ("md5", "sha1", "sha256")
    show_progress : bool, default=True
        Whether to show download progress bar
    create_dirs : bool, default=True
        Whether to create parent directories if they don't exist

    Returns
    -------
    Path
        Path object pointing to the downloaded file

    Raises
    ------
    ValueError
        If URL is invalid or checksum verification fails
    FileExistsError
        If file exists and overwrite=False
    requests.RequestException
        If download fails after all retries

    Examples
    --------
    Basic usage:
    >>> file_path = download_origin_data(
    ...     "https://example.com/data.csv",
    ...     "data/raw_data.csv"
    ... )
    >>> print(f"Downloaded to: {file_path}")
    Downloaded to: data/raw_data.csv

    With checksum verification:
    >>> file_path = download_origin_data(
    ...     "https://example.com/important_data.xlsx",
    ...     "data/important_data.xlsx",
    ...     expected_checksum="5d41402abc4b2a76b9719d911017c592",
    ...     checksum_algorithm="md5"
    ... )

    With custom headers and retry settings:
    >>> headers = {"User-Agent": "MyApp/1.0"}
    >>> file_path = download_origin_data(
    ...     "https://api.example.com/dataset.json",
    ...     "data/dataset.json",
    ...     headers=headers,
    ...     max_retries=5,
    ...     retry_delay=2.0
    ... )

    Download without progress bar:
    >>> file_path = download_origin_data(
    ...     "https://example.com/data.tsv",
    ...     "data/data.tsv",
    ...     show_progress=False,
    ...     overwrite=True
    ... )
    """
    # Convert to Path object
    local_path = Path(local_path)

    # Validate URL
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: {url}")

    tqdm.write(f"Downloading data from {url}...")
    tqdm.write(f"Target location: {local_path}")

    # Check if file exists
    if local_path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {local_path}. Use overwrite=True to replace it."
        )

    # Create parent directories if needed
    if create_dirs and local_path.parent != Path("."):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tqdm.write(f"Created directory: {local_path.parent}")

    # Prepare headers
    default_headers = {"User-Agent": "tidymut-data-downloader/1.0"}
    if headers:
        default_headers.update(headers)

    # Download with retries
    for attempt in range(max_retries):
        try:
            tqdm.write(f"Download attempt {attempt + 1}/{max_retries}...")

            # Make initial request to get file size
            response = requests.head(
                url,
                headers=default_headers,
                timeout=timeout,
                verify=verify_ssl,
                allow_redirects=True,
            )
            response.raise_for_status()

            # Get content length for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Start actual download
            response = requests.get(
                url,
                headers=default_headers,
                timeout=timeout,
                verify=verify_ssl,
                stream=True,
                allow_redirects=True,
            )
            response.raise_for_status()

            # Initialize progress bar
            progress_bar = None
            if show_progress and total_size > 0:
                progress_bar = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {local_path.name}",
                )

            # Initialize checksum if needed
            checksum_hash = None
            if expected_checksum:
                if checksum_algorithm.lower() == "md5":
                    checksum_hash = hashlib.md5()
                elif checksum_algorithm.lower() == "sha1":
                    checksum_hash = hashlib.sha1()
                elif checksum_algorithm.lower() == "sha256":
                    checksum_hash = hashlib.sha256()
                else:
                    raise ValueError(
                        f"Unsupported checksum algorithm: {checksum_algorithm}"
                    )

            # Download and save file
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        if checksum_hash:
                            checksum_hash.update(chunk)
                        if progress_bar:
                            progress_bar.update(len(chunk))

            if progress_bar:
                progress_bar.close()

            # Verify checksum if provided
            if expected_checksum and checksum_hash:
                calculated_checksum = checksum_hash.hexdigest()
                if calculated_checksum.lower() != expected_checksum.lower():
                    local_path.unlink()  # Remove corrupted file
                    raise ValueError(
                        f"Checksum verification failed. "
                        f"Expected: {expected_checksum}, "
                        f"Got: {calculated_checksum}"
                    )
                tqdm.write(
                    f"Checksum verification passed ({checksum_algorithm.upper()})"
                )

            file_size = local_path.stat().st_size
            tqdm.write(f"Successfully downloaded {file_size:,} bytes to {local_path}")
            return local_path

        except requests.RequestException as e:
            tqdm.write(f"Download attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                tqdm.write(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                tqdm.write("All download attempts failed.")
                raise

        except Exception as e:
            # Clean up partial download on unexpected errors
            if local_path.exists():
                local_path.unlink()
            raise
    # should never reached because already deal with exception in loop
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.")


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

    Example
    -------
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
        Number of parallel workers for processing, set to -1 for all available CPUs

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
