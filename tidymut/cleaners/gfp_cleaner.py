# tidymut/cleaners/gfp_cleaner.py
from __future__ import annotations

import pandas as pd
from typing import TYPE_CHECKING
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
    average_labels_by_name,
    convert_to_mutation_dataset_format,
    add_column,
)
from ..core.dataset import MutationDataset
from ..core.pipeline import Pipeline, create_pipeline

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

__all__ = [
    "GFPCleanerConfig",
    "create_GFP_cleaner",
    "clean_GFP_dataset",
]


def __dir__() -> List[str]:
    return __all__


# Create module logger
logger = logging.getLogger(__name__)


@dataclass
class GFPCleanerConfig(BaseCleanerConfig):

    # Column mapping configuration
    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "mutated_sequence": "mut_seq",
            "mutant": "mut_info",
            "DMS_score": "label",
        }
    )

    # Data filtering configuration
    filters: Dict[str, Callable] = field(
        default_factory=lambda: {
            "label": lambda s: pd.to_numeric(s, errors="coerce").notna()
        }
    )

    # Type conversion configuration
    type_conversions: Dict[str, str] = field(default_factory=lambda: {"label": "float"})

    # Mutation validation parameters
    validate_mut_workers: int = 16

    # Wildtype validation parameters
    validate_wt_workers: int = 16

    # Score columns configuration
    label_columns: List[str] = field(default_factory=lambda: ["label"])
    primary_label_column: str = "label"

    # Override default pipeline name
    pipeline_name: str = "GFP pipeline"

    def validate(self) -> None:
        """Validate GFP-specific configuration parameters

        Raises
        ------
        ValueError
            If configuration is invalid
        """
        # Call parent validation
        super().validate()

        # Validate score columns
        if not self.label_columns:
            raise ValueError("label_columns cannot be empty")

        if self.primary_label_column not in self.label_columns:
            raise ValueError(
                f"primary_label_column '{self.primary_label_column}' "
                f"must be in label_columns {self.label_columns}"
            )

        # Validate column mapping
        required_mappings = {"mutated_sequence", "mutant", "DMS_score"}
        missing = required_mappings - set(self.column_mapping.keys())
        if missing:
            raise ValueError(f"Missing required column mappings: {missing}")


def create_GFP_cleaner(
    dataset_or_path: Optional[Union[pd.DataFrame, str, Path]] = None,
    config: Optional[Union[GFPCleanerConfig, Dict[str, Any], str, Path]] = None,
) -> Pipeline:
    """Create GFP dataset cleaning pipeline

    Parameters
    ----------
    dataset_or_path : Optional[Union[pd.DataFrame, str, Path]], default=None
        Raw dataset DataFrame or file path to GFP dataset.
    config : Optional[Union[GFPCleanerConfig, Dict[str, Any], str, Path]]
        Configuration for the cleaning pipeline. Can be:
        - GFPCleanerConfig object
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
    """
    # Handle configuration parameter
    if config is None:
        final_config = GFPCleanerConfig()
    elif isinstance(config, GFPCleanerConfig):
        final_config = config
    elif isinstance(config, dict):
        # Partial configuration - merge with defaults
        default_config = GFPCleanerConfig()
        final_config = default_config.merge(config)
    elif isinstance(config, (str, Path)):
        # Load from file
        final_config = GFPCleanerConfig.from_json(config)
    else:
        raise TypeError(
            f"config must be GFPCleanerConfig, dict, str, Path or None, "
            f"got {type(config)}"
        )

    # Log configuration summary
    logger.info(
        f"GFP dataset will be cleaned with pipeline: {final_config.pipeline_name}"
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
                add_column,
                dataset_name="gfp",
                column_name="name",
            )
            .delayed_then(
                validate_mutations,
                mutation_column=final_config.column_mapping.get("mutant", "mutant"),
                mutation_sep=":",
                is_zero_based=False,
                num_workers=final_config.validate_mut_workers,
            )
            .delayed_then(
                infer_wildtype_sequences,
                mutation_column=final_config.column_mapping.get("mutant", "mutant"),
                sequence_column=final_config.column_mapping.get(
                    "mutated_sequence", "mutated_sequence"
                ),
                is_zero_based=True,
                num_workers=final_config.validate_wt_workers,
            )
            .delayed_then(
                average_labels_by_name,
                name_columns=final_config.column_mapping.get("mutant", "mutant"),
                label_columns=final_config.label_columns,
            )
            .delayed_then(
                convert_to_mutation_dataset_format,
                name_column="name",
                mutation_column=final_config.column_mapping.get("mutant", "mutant"),
                mutated_sequence_column=final_config.column_mapping.get(
                    "mutated_sequence", "mutated_sequence"
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
        logger.error(f"Error in creating GFP cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in creating GFP cleaning pipeline: {str(e)}")


def clean_GFP_dataset(
    pipeline: Pipeline,
) -> Tuple[Pipeline, MutationDataset]:
    """Clean GFP dataset using configurable pipeline

    Parameters
    ----------
    pipeline : Pipeline
        GFP dataset cleaning pipeline

    Returns
    -------
    Tuple[Pipeline, MutationDataset]
        - Pipeline: The cleaned pipeline
        - MutationDataset: The cleaned GFP dataset

    Examples
    --------
    >>> pipeline = create_gfp_cleaner(df)  # df is raw GFP dataset file
    Use default configuration:

    >>> pipeline, dataset = clean_GFP_dataset(pipeline)

    Use partial configuration:

    >>> pipeline, dataset = clean_GFP_dataset(df, config={
    ...     "validate_mut_workers": 8,
    ... })

    Load configuration from file:

    >>> pipeline, dataset = clean_GFP_dataset(df, config="config.json")
    """
    try:
        # Run pipeline
        pipeline.execute()

        # Extract results
        gfp_dataset_df, gfp_ref_seq = pipeline.data
        gfp_dataset = MutationDataset.from_dataframe(gfp_dataset_df, gfp_ref_seq)

        logger.info(
            f"Successfully cleaned GFP dataset: "
            f"{len(gfp_dataset_df)} mutations from {len(gfp_ref_seq)} proteins"
        )

        return pipeline, gfp_dataset
    except Exception as e:
        logger.error(f"Error in running GFP dataset cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in running GFP dataset cleaning pipeline: {str(e)}")
