# tidymut/cleaners/ConFit_cleaner.py
from __future__ import annotations

import pandas as pd
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .base_config import BaseCleanerConfig
from .basic_cleaners import (
    extract_and_rename_columns,
    filter_and_clean_data,
    convert_data_types,
    validate_mutations,
    convert_to_mutation_dataset_format,
)
from .ConFit_custom_cleaners import read_ConFit_data, validate_mutations_and_sequences
from ..core.dataset import MutationDataset
from ..core.pipeline import Pipeline, create_pipeline

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple, Union, Callable

__all__ = [
    "ConFitCleanerConfig",
    "create_ConFit_cleaner",
    "clean_ConFit_dataset",
]


def __dir__() -> List[str]:
    return __all__


# Create module logger
logger = logging.getLogger(__name__)


@dataclass
class ConFitCleanerConfig(BaseCleanerConfig):

    # Column mapping configuration
    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "seq": "mut_seq",
            "log_fitness": "label",
            "mutant": "mut_info",
            "name": "name",
            "wt_seq": "wt_seq",
        }
    )

    # Data filtering configuration
    filters: Dict[str, Callable] = field(
        default_factory=lambda: {
            "label": lambda s: pd.to_numeric(s, errors="coerce").notna()
        }
    )

    # Type conversion configuration
    type_conversions: Dict[str, str] = field(
        default_factory=lambda: {"label": "float"}
    )

    # Mutation validation parameters
    validation_workers: int = 16

    # Score columns configuration
    label_columns: List[str] = field(default_factory=lambda: ["label"])
    primary_label_column: str = "label"

    # Override default pipeline name
    pipeline_name: str = "ConFit Cleaning Pipeline"

    def validate(self) -> None:
        """Validate ProteinGym-specific configuration parameters

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
        required_mappings = {"seq", "log_fitness", "mutant"}
        missing = required_mappings - set(self.column_mapping.keys())
        if missing:
            raise ValueError(f"Missing required column mappings: {missing}")


def create_ConFit_cleaner(
    data_path: Union[str, Path],
    config: Optional[Union[ConFitCleanerConfig, Dict[str, Any], str, Path]] = None,
) -> Pipeline:
    """Create ConFit dataset cleaning pipeline

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to directory containing ConFit CSV files
    config : Optional[Union[ConFitCleanerConfig, Dict[str, Any], str, Path]]
        Configuration for the cleaning pipeline. Can be:
        - ConFitCleanerConfig object
        - Dictionary with configuration parameters (merged with defaults)
        - Path to JSON configuration file (str or Path)
        - None (uses default configuration)

    Returns
    -------
    Pipeline
        The cleaning pipeline

    Raises
    ------
    TypeError
        If config has invalid type
    ValueError
        If configuration validation fails

    Examples
    --------
    Process directory of ConFit CSV files:

    >>> pipeline = create_ConFit_cleaner("ConFit_data/")
    >>> pipeline, dataset = clean_ConFit_dataset(pipeline)

    Process zip file:

    >>> pipeline = create_ConFit_cleaner("ConFit_data/")
    >>> pipeline, dataset = clean_ConFit_dataset(pipeline)

    Custom configuration:

    >>> config = {
    ...     "validation_workers": 8,
    ...     "handle_multiple_wt": "first"
    ... }
    >>> pipeline = create_ConFit_cleaner("data/", config=config)

    Load configuration from file:

    >>> pipeline = create_ConFit_cleaner("data/", config="config.json")
    """
    # Validate input path
    path_obj = Path(data_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    # ConFit only supports directory
    if not path_obj.is_dir():
        raise TypeError(
            f"ConFit cleaner only supports directory input, "
            f"got: {data_path}"
        )

    # Handle configuration parameter
    if config is None:
        final_config = ConFitCleanerConfig()
    elif isinstance(config, ConFitCleanerConfig):
        final_config = config
    elif isinstance(config, dict):
        # Partial configuration - merge with defaults
        default_config = ConFitCleanerConfig()
        final_config = default_config.merge(config)
    elif isinstance(config, (str, Path)):
        # Load from file
        final_config = ConFitCleanerConfig.from_json(config)
    else:
        raise TypeError(
            f"config must be ConFitCleanerConfig, dict, str, Path or None, "
            f"got {type(config)}"
        )

    # Log configuration summary
    logger.info(
        f"ConFit dataset will be cleaned with pipeline: {final_config.pipeline_name}"
    )
    logger.debug(f"Configuration:\n{final_config.get_summary()}")

    try:
        # Create pipeline
        pipeline = create_pipeline(data_path, final_config.pipeline_name)

        # Add cleaning steps using basic_cleaners functions
        pipeline = (
            pipeline.delayed_then(
                read_ConFit_data,
            )
            .delayed_then(
                extract_and_rename_columns,
                column_mapping=final_config.column_mapping,
            )
            .delayed_then(
                filter_and_clean_data,
                filters=final_config.filters,
            )
            .delayed_then(
                convert_data_types,
                type_conversions=final_config.type_conversions,
            )
            .delayed_then(
                validate_mutations,
                mutation_column=final_config.column_mapping.get("mutant", "mutant"),
                mutation_sep=",",
                is_zero_based=False,
                num_workers=final_config.validation_workers,
            )
            .delayed_then(
                validate_mutations_and_sequences,
                wt_sequence_column=final_config.column_mapping.get("wt_seq", "wt_seq"),
                name_column=final_config.column_mapping.get("name", "name"),
                mutation_column=final_config.column_mapping.get("mutant", "mutant"),
                mut_sequence_column=final_config.column_mapping.get("seq", "seq"),
                mutation_sep=",",
                is_zero_based=True,
                num_workers=final_config.validation_workers,
            )
            .delayed_then(
                convert_to_mutation_dataset_format,
                name_column=final_config.column_mapping.get("name", "name"),
                mutation_column=final_config.column_mapping.get("mutant", "mutant"),
                sequence_column=final_config.column_mapping.get("wt_seq", "wt_seq"),
                mutated_sequence_column=final_config.column_mapping.get("seq", "seq"),
                label_column=final_config.primary_label_column,
                is_zero_based=True,
            )
        )

        return pipeline

    except Exception as e:
        logger.error(f"Error in creating ConFit cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in creating ConFit cleaning pipeline: {str(e)}")


def clean_ConFit_dataset(pipeline: Pipeline) -> Tuple[Pipeline, MutationDataset]:
    """Clean ConFit dataset using configurable pipeline

    Parameters
    ----------
    pipeline : Pipeline
        ConFit dataset cleaning pipeline

    Returns
    -------
    Tuple[Pipeline, MutationDataset]
        - Pipeline: The cleaned pipeline
        - MutationDataset: The cleaned ConFit dataset
    """
    try:
        # Run pipeline
        pipeline.execute()

        # Extract results
        confit_dataset_df, confit_ref_seq = pipeline.data
        confit_dataset = MutationDataset.from_dataframe(
            confit_dataset_df, confit_ref_seq
        )

        logger.info(
            f"Successfully cleaned ConFit dataset: "
            f"{len(confit_dataset_df)} mutations from {len(confit_ref_seq)} proteins"
        )

        return pipeline, confit_dataset

    except Exception as e:
        logger.error(f"Error in running ConFit cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in running ConFit cleaning pipeline: {str(e)}")