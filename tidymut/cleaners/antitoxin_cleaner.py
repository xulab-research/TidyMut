# tidymut/cleaners/antitoxin_cleaner.py
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
    validate_mutations,
    add_column,
    apply_mutations_to_sequences,
    average_labels_by_name,
)
from .coves_custom_cleaners import (
    add_wild_type_sequence,
    simplify_mutations,
)
from .cdna_proteolysis_custom_cleaners import subtract_labels_by_wt

from ..core.dataset import MutationDataset
from ..core.pipeline import Pipeline, create_pipeline

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

__all__ = [
    "class AntitoxinCleanerConfig",
    "create_aav_capsid_cleaner",
    "clean_aav_capsid_dataset",
]


def __dir__() -> List[str]:
    return __all__


# Create module logger
logger = logging.getLogger(__name__)


@dataclass
class AntitoxinCleanerConfig(BaseCleanerConfig):

    # Column mapping configuration
    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "mutation": "mut_info",
            "label": "label",
        }
    )

    # Data filtering configuration
    filters: Dict[str, Callable] = field(
        default_factory=lambda: {
            "label": lambda s: pd.to_numeric(s, errors="coerce").notna()
        }
    )

    # obtained from the article
    wt_sequence = "MANVEKMSVAVTPQQAAVMREAVEAGEYATASEIVREAVRDWLAKRELRHDDIRRLRQLWDEGKASGRPEPVDFDALRKEARQKLTEVPPNGR"
    # Type conversion configuration
    type_conversions: Dict[str, str] = field(default_factory=lambda: {"label": "float"})

    # Mutation validation parameters
    validate_mut_workers: int = 16

    process_workers: int = 16

    # Score columns configuration
    label_columns: List[str] = field(default_factory=lambda: ["label"])
    primary_label_column: str = "label"

    # Override default pipeline name
    pipeline_name: str = "Antitoxin Pipeline"

    def validate(self) -> None:
        """Validate cDNAProteolysis-specific configuration parameters

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
        required_mappings = {"mutation"}
        missing = required_mappings - set(self.column_mapping.keys())
        if missing:
            raise ValueError(f"Missing required column mappings: {missing}")


def create_antitoxin_cleaner(
    dataset_or_path: Optional[Union[pd.DataFrame, str, Path]] = None,
    config: Optional[Union[AntitoxinCleanerConfig, Dict[str, Any], str, Path]] = None,
) -> Pipeline:
    # Handle configuration parameter
    if config is None:
        final_config = AntitoxinCleanerConfig()
    elif isinstance(config, AntitoxinCleanerConfig):
        final_config = config
    elif isinstance(config, dict):
        # Partial configuration - merge with defaults
        default_config = AntitoxinCleanerConfig()
        final_config = default_config.merge(config)
    elif isinstance(config, (str, Path)):
        # Load from file
        final_config = AntitoxinCleanerConfig.from_json(config)
    else:
        raise TypeError(
            f"config must be AntitoxinCleanerConfig, dict, str, Path or None, "
            f"got {type(config)}"
        )

    # Log configuration summary
    logger.info(
        f"Antitoxin dataset will be cleaned with pipeline: {final_config.pipeline_name}"
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
            .delayed_then(
                filter_and_clean_data,
                filters=final_config.filters,
            )
            .delayed_then(
                convert_data_types,
                type_conversions=final_config.type_conversions,
            )
            .delayed_then(add_column, dataset_name="antitoxin", column_name="name")
            .delayed_then(
                add_wild_type_sequence,
                wt_sequence_column="wt_seq",
                wt_sequence=final_config.wt_sequence,
            )
            .delayed_then(
                simplify_mutations,
                mutation_column=final_config.column_mapping.get("mutation", "mutation"),
                mutation_sep=":",
            )
            .delayed_then(
                validate_mutations,
                mutation_column=final_config.column_mapping.get("mutation", "mutation"),
                mutation_sep=",",
                is_zero_based=True,
                exclude_patterns="WT",
                num_workers=final_config.validate_mut_workers,
            )
            .delayed_then(
                average_labels_by_name,
                name_columns=(
                    "name",
                    final_config.column_mapping.get("mutation", "mutation"),
                ),
                label_columns=final_config.primary_label_column,
            )
            .delayed_then(
                subtract_labels_by_wt,
                name_column="name",
                label_columns=final_config.primary_label_column,
                mutation_column=final_config.column_mapping.get("mutation", "mutation"),
                wt_identifier="WT",
                in_place=True,
            )
            .delayed_then(
                apply_mutations_to_sequences,
                sequence_column="wt_seq",
                name_column="name",
                mutation_column=final_config.column_mapping.get("mutation", "mutation"),
                is_zero_based=True,
                sequence_type="protein",
                num_workers=final_config.process_workers,
            )
            .delayed_then(
                convert_to_mutation_dataset_format,
                name_column="name",
                mutation_column=final_config.column_mapping.get("mutation", "mutation"),
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
        logger.error(f"Error in creating aav capsid cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in creating aav capsid cleaning pipeline: {str(e)}")


def clean_antitoxin_dataset(
    pipeline: Pipeline,
) -> Tuple[Pipeline, MutationDataset]:
    try:
        # Run pipeline
        pipeline.execute()

        # Extract results
        antitoxin_dataset_df, antitoxin_ref_seq = pipeline.data
        antitoxin_dataset = MutationDataset.from_dataframe(
            antitoxin_dataset_df, antitoxin_ref_seq
        )

        logger.info(
            f"Successfully cleaned antitoxin dataset: "
            f"{len(antitoxin_dataset_df)} mutations from {len(antitoxin_ref_seq)} proteins"
        )

        return pipeline, antitoxin_dataset
    except Exception as e:
        logger.error(f"Error in running antitoxin dataset cleaning pipeline: {str(e)}")
        raise RuntimeError(
            f"Error in running antitoxin dataset cleaning pipeline: {str(e)}"
        )
