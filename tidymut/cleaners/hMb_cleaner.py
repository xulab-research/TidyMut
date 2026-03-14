# tidymut/cleaners/gb_cleaner.py
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
    convert_to_mutation_dataset_format,
    add_column,
    average_labels_by_name,
    validate_mutations,
    apply_mutations_to_sequences,
)
from .hMb_custom_cleaners import convert_codon_to_amino_acid

from ..core.dataset import MutationDataset
from ..core.pipeline import Pipeline, create_pipeline

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

__all__ = [
    "hMbCleanerConfig",
    "create_hMb_cleaner",
    "clean_hMb_dataset",
]


def __dir__() -> List[str]:
    return __all__


# Create module logger
logger = logging.getLogger(__name__)


@dataclass
class hMbCleanerConfig(BaseCleanerConfig):

    # Column mapping configuration
    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "COD": "codon_mutations",
            "fitness": "label",
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

    # obtained from the article
    wt_sequence = "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHGATVLTALGGILKKKGHHEAEIKPLAQSHATKHKIPVKYLEFISECIIQVLQSKHPGDFGADAQGAMNKALELFRKDMASNYKELGFQG"

    # Mutation validation parameters
    validate_mut_workers: int = 16

    # Processing parameters
    process_workers: int = 16

    # Score columns configuration
    label_columns: List[str] = field(default_factory=lambda: ["label"])
    primary_label_column: str = "label"

    # Override default pipeline name
    pipeline_name: str = "hMb Cleaning Pipeline"

    def validate(self) -> None:
        """Validate hMb-specific configuration parameters

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
        required_mappings = {"COD", "fitness"}
        missing = required_mappings - set(self.column_mapping.keys())
        if missing:
            raise ValueError(f"Missing required column mappings: {missing}")


def create_hMb_cleaner(
    dataset_or_path: Optional[Union[pd.DataFrame, str, Path]] = None,
    config: Optional[Union[hMbCleanerConfig, Dict[str, Any], str, Path]] = None,
) -> Pipeline:
    # Handle configuration parameter
    if config is None:
        final_config = hMbCleanerConfig()
    elif isinstance(config, hMbCleanerConfig):
        final_config = config
    elif isinstance(config, dict):
        # Partial configuration - merge with defaults
        default_config = hMbCleanerConfig()
        final_config = default_config.merge(config)
    elif isinstance(config, (str, Path)):
        # Load from file
        final_config = hMbCleanerConfig.from_json(config)
    else:
        raise TypeError(
            f"config must be hMbCleanerConfig, dict, str, Path or None, "
            f"got {type(config)}"
        )

    # Log configuration summary
    logger.info(
        f"hMb dataset will be cleaned with pipeline: {final_config.pipeline_name}"
    )
    logger.debug(f"Configuration:\n{final_config.get_summary()}")

    try:
        # Create cleaning pipeline
        pipeline = create_pipeline(dataset_or_path, final_config.pipeline_name)

        # Add cleaning steps
        pipeline = (
            pipeline.delayed_then(
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
            .delayed_then(add_column, dataset_name="hMb", column_name="name")
            .delayed_then(
                add_column,
                column_name="wt_seq",
                dataset_name=final_config.wt_sequence,
            )
            .delayed_then(
                convert_codon_to_amino_acid,
                codon_column="codon_mutations",
                amino_acid_column="mut_info",
                drop_codon_column=True,
            )
            .delayed_then(
                validate_mutations,
                mutation_column="mut_info",
                num_workers=final_config.validate_mut_workers,
            )
            .delayed_then(
                apply_mutations_to_sequences,
                sequence_column="wt_seq",
                num_workers=final_config.process_workers,
            )
            .delayed_then(
                average_labels_by_name,
                name_columns="mut_info",
                label_columns=final_config.primary_label_column,
            )
            .delayed_then(
                convert_to_mutation_dataset_format,
                name_column="name",
                mutation_column="mut_info",
                sequence_column="wt_seq",
                label_column=final_config.primary_label_column,
                is_zero_based=True,
            )
        )

        # Create pipeline based on dataset_or_path type
        if isinstance(dataset_or_path, (str, Path)):
            pipeline.add_delayed_step(read_dataset, 0, file_format="csv")
        elif not isinstance(dataset_or_path, pd.DataFrame):
            raise TypeError(
                f"dataset_or_path must be pd.DataFrame or str/Path, "
                f"got {type(dataset_or_path)}"
            )

        return pipeline

    except Exception as e:
        logger.error(f"Error in creating hMb cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in creating hMb cleaning pipeline: {str(e)}")


def clean_hMb_dataset(
    pipeline: Pipeline,
) -> Tuple[Pipeline, MutationDataset]:
    try:
        # Run pipeline
        pipeline.execute()

        # Extract results
        hMb_dataset_df, hMb_ref_seq = pipeline.data
        hMb_dataset = MutationDataset.from_dataframe(hMb_dataset_df, hMb_ref_seq)

        logger.info(
            f"Successfully cleaned hMb dataset: "
            f"{len(hMb_dataset_df)} mutations from {len(hMb_ref_seq)} proteins"
        )

        return pipeline, hMb_dataset
    except Exception as e:
        logger.error(f"Error in running hMb dataset cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in running hMb dataset cleaning pipeline: {str(e)}")
