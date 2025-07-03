# tidymut/cleaners/k50_cleaner.py
from __future__ import annotations

import pandas as pd
from typing import Tuple, Dict, Any, Optional, Union, Callable, List, Literal
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .base_config import BaseCleanerConfig
from .basic_cleaners import (
    read_dataset,
    extract_and_rename_columns,
    filter_and_clean_data,
    convert_data_types,
    validate_mutations,
    infer_wildtype_sequences,
    convert_to_mutation_dataset_format,
)
from ..core.dataset import MutationDataset
from ..core.pipeline import Pipeline, create_pipeline

__all__ = ["K50CleanerConfig", "create_k50_cleaner", "clean_k50_dataset"]


def __dir__() -> List[str]:
    return __all__


# Create module logger
logger = logging.getLogger(__name__)


@dataclass
class K50CleanerConfig(BaseCleanerConfig):
    """Configuration class for K50 dataset cleaner

    Inherits from BaseCleanerConfig and adds K50-specific configuration options.

    Raw K50 dataset DataFrame or file path to K50 dataset
    - Download from: https://zenodo.org/records/7992926
    - File: `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` in `Processed_K50_dG_datasets.zip`

    Attributes
    ----------
    column_mapping : Dict[str, str]
        Mapping from source to target column names
    filters : Dict[str, Callable]
        Filter conditions for data cleaning
    type_conversions : Dict[str, str]
        Data type conversion specifications
    validation_workers : int
        Number of workers for mutation validation, set to -1 to use all available CPUs
    infer_wt_workers : int
        Number of workers for wildtype sequence inference, set to -1 to use all available CPUs
    handle_multiple_wt : Literal["error", "first", "separate"], default="error"
        Strategy for handling multiple wildtype sequences ('error', 'first', 'separate')
    label_columns : List[str]
        List of score columns to process
    primary_label_column : str
        Primary score column for the dataset
    """

    # Column mapping configuration
    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "WT_name": "name",
            "aa_seq": "mut_seq",
            "mut_type": "mut_info",
            "ddG_ML": "ddG",
        }
    )

    # Data filtering configuration
    filters: Dict[str, Callable] = field(
        default_factory=lambda: {"ddG": lambda x: x != "-"}
    )

    # Type conversion configuration
    type_conversions: Dict[str, str] = field(default_factory=lambda: {"ddG": "float"})

    # Mutation validation parameters
    validation_workers: int = 16

    # Wildtype inference parameters
    infer_wt_workers: int = 16
    handle_multiple_wt: Literal["error", "first", "separate"] = "error"

    # Score columns configuration
    label_columns: List[str] = field(default_factory=lambda: ["ddG"])
    primary_label_column: str = "ddG"

    # Override default pipeline name
    pipeline_name: str = "k50_cleaner"

    def validate(self) -> None:
        """Validate K50-specific configuration parameters

        Raises
        ------
        ValueError
            If configuration is invalid
        """
        # Call parent validation
        super().validate()

        # Validate handle_multiple_wt parameter
        valid_strategies = ["error", "first", "separate"]
        if self.handle_multiple_wt not in valid_strategies:
            raise ValueError(
                f"handle_multiple_wt must be one of {valid_strategies}, "
                f"got {self.handle_multiple_wt}"
            )

        # Validate score columns
        if not self.label_columns:
            raise ValueError("label_columns cannot be empty")

        if self.primary_label_column not in self.label_columns:
            raise ValueError(
                f"primary_label_column '{self.primary_label_column}' "
                f"must be in label_columns {self.label_columns}"
            )

        # Validate column mapping
        required_mappings = {"WT_name", "aa_seq", "mut_type"}
        missing = required_mappings - set(self.column_mapping.keys())
        if missing:
            raise ValueError(f"Missing required column mappings: {missing}")


def create_k50_cleaner(
    dataset_or_path: Union[pd.DataFrame, str, Path],
    config: Optional[Union[K50CleanerConfig, Dict[str, Any], str, Path]] = None,
) -> Pipeline:
    """Create K50 dataset cleaning pipeline

    Parameters
    ----------
    dataset_or_path : Union[pd.DataFrame, str, Path]
        Raw K50 dataset DataFrame or file path to K50 dataset
        - Download from: https://zenodo.org/records/7992926
        - File: `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` in `Processed_K50_dG_datasets.zip`
    config : Optional[Union[K50CleanerConfig, Dict[str, Any], str, Path]]
        Configuration for the cleaning pipeline. Can be:
        - K50CleanerConfig object
        - Dictionary with configuration parameters (merged with defaults)
        - Path to JSON configuration file (str or Path)
        - None (uses default configuration)

    Returns
    -------
    Pipeline
        Pipeline: The cleaning pipeline used

    Raises
    ------
    TypeError
        If config has invalid type
    ValueError
        If configuration validation fails

    Examples
    --------
    Use default configuration:
    >>> pipeline, dataset = clean_k50_dataset(df)

    Use partial configuration:
    >>> pipeline, dataset = clean_k50_dataset(df, config={
    ...     "validation_workers": 8,
    ...     "handle_multiple_wt": "first"
    ... })

    Load configuration from file:
    >>> pipeline, dataset = clean_k50_dataset(df, config="config.json")
    """
    # Handle configuration parameter
    if config is None:
        final_config = K50CleanerConfig()
    elif isinstance(config, K50CleanerConfig):
        final_config = config
    elif isinstance(config, dict):
        # Partial configuration - merge with defaults
        default_config = K50CleanerConfig()
        final_config = default_config.merge(config)
    elif isinstance(config, (str, Path)):
        # Load from file
        final_config = K50CleanerConfig.from_json(config)
    else:
        raise TypeError(
            f"config must be K50CleanerConfig, dict, str, Path or None, "
            f"got {type(config)}"
        )

    # Log configuration summary
    logger.info(
        f"K50 dataset will cleaning with pipeline: {final_config.pipeline_name}"
    )
    logger.debug(f"Configuration:\n{final_config.get_summary()}")

    try:
        # Create pipeline
        pipeline = create_pipeline(dataset_or_path, final_config.pipeline_name)

        # Add cleaning steps
        pipeline = (
            pipeline.delayed_then(
                extract_and_rename_columns,
                column_mapping=final_config.column_mapping,
            )
            .delayed_then(filter_and_clean_data, filters=final_config.filters)
            .delayed_then(
                convert_data_types, type_conversions=final_config.type_conversions
            )
            .delayed_then(
                validate_mutations,
                mutation_column=final_config.column_mapping.get("mut_type", "mut_type"),
                mutation_sep="_",
                is_zero_based=False,
                num_workers=final_config.validation_workers,
            )
            .delayed_then(
                infer_wildtype_sequences,
                label_columns=final_config.label_columns,
                handle_multiple_wt=final_config.handle_multiple_wt,
                is_zero_based=True,  # Always True after validate_mutations
                num_workers=final_config.infer_wt_workers,
            )
            .delayed_then(
                convert_to_mutation_dataset_format,
                name_column=final_config.column_mapping.get("WT_name", "WT_name"),
                mutation_column=final_config.column_mapping.get("mut_type", "mut_type"),
                mutated_sequence_column=final_config.column_mapping.get(
                    "aa_seq", "aa_seq"
                ),
                label_column=final_config.primary_label_column,
                is_zero_based=True,
            )
        )

        # Create pipeline based on dataset_or_path type
        if isinstance(dataset_or_path, (str, Path)):
            pipeline.add_delayed_step(read_dataset, 0)
        elif not isinstance(dataset_or_path, pd.DataFrame):
            raise TypeError(
                f"dataset_or_path must be pd.DataFrame or str/Path, "
                f"got {type(dataset_or_path)}"
            )

        return pipeline

    except Exception as e:
        logger.error(f"Error in creating K50 cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in creating K50 cleaning pipeline: {str(e)}")


def clean_k50_dataset(pipeline: Pipeline) -> Tuple[Pipeline, MutationDataset]:
    """Clean K50 dataset using configurable pipeline

    Parameters
    ----------
    pipeline : Pipeline
        K50 dataset cleaning pipeline

    Returns
    -------
    Tuple[Pipeline, MutationDataset]
        - Pipeline: The cleaned pipeline
        - MutationDataset: The cleaned K50 dataset
    """
    try:
        # Run pipeline
        pipeline.execute()

        # Extract results
        k50_dataset_df, k50_ref_seq = pipeline.data
        k50_dataset = MutationDataset.from_dataframe(k50_dataset_df, k50_ref_seq)

        logger.info(
            f"Successfully cleaned K50 dataset: "
            f"{len(k50_dataset_df)} mutations from {len(k50_ref_seq)} proteins"
        )

        return pipeline, k50_dataset
    except Exception as e:
        logger.error(f"Error in running K50 cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in running K50 cleaning pipeline: {str(e)}")
