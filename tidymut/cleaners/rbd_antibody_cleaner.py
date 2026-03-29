from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tidymut.cleaners.base_config import BaseCleanerConfig
from tidymut.cleaners.basic_cleaners import (
    add_column,
    average_labels_by_name,
    convert_data_types,
    convert_to_mutation_dataset_format,
    extract_and_rename_columns,
    filter_and_clean_data,
    read_dataset,
    validate_mutations,
    subtract_labels_by_wt,
)
from tidymut.cleaners.rbd_custom_cleaners import (
    apply_mutations_preserving_wild_type,
    mark_wild_type_by_variant_class,
)
from tidymut.core.dataset import MutationDataset
from tidymut.core.pipeline import Pipeline, create_pipeline

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# Create module logger
logger = logging.getLogger(__name__)

__all__ = [
    "RBDAntibodyCleanerConfig",
    "create_rbd_antibody_cleaner",
    "clean_rbd_antibody_dataset",
]


def __dir__() -> List[str]:
    return __all__


@dataclass
class RBDAntibodyCleanerConfig(BaseCleanerConfig):
    """
    Configuration class for the RBD antibody dataset cleaner.

    This cleaner is designed for tables like
    `SARS-CoV-2-RBD_MAP_AZ_Abs_scores.csv`, where:
    - `aa_substitutions` stores 1-based amino-acid mutations separated by spaces
    - `score` is the negative-log-transformed per-barcode escape score
    - higher `score` values indicate weaker escape, reflecting better binding capacity
    - identical `(name, aa_substitutions)` observations should be averaged

    Attributes
    ----------
    column_mapping : Dict[str, str]
        Mapping from raw antibody table columns to tidymut standard columns.
    filters : Dict[str, Callable]
        Filter conditions used to remove rows that fail score or QC checks.
    drop_na_columns : List[str]
        Required columns that must be present before downstream processing.
    type_conversions : Dict[str, str]
        Data type conversion specifications for score and mutation counts.
    wt_sequence : str
        Reference RBD sequence shared by the antibody datasets.
    label_columns : List[str]
        Label columns available in the cleaned dataset.
    primary_label_column : str
        Primary label column used when building the final MutationDataset.
    validate_mut_workers : int
        Number of workers used for mutation validation.
    process_workers : int
        Number of workers used for applying mutations to the RBD sequence.
    pipeline_name : str
        Name of the cleaning pipeline.
    """

    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "aa_substitutions": "aa_substitutions",
            "score": "label",
            "name": "name",
            "variant_class": "variant_class",
            "pass_pre_count_filter": "pass_pre_count_filter",
            "pass_ACE2bind_expr_filter": "pass_ACE2bind_expr_filter",
        }
    )

    filters: Dict[str, Callable] = field(
        default_factory=lambda: {
            "label": lambda s: pd.to_numeric(s, errors="coerce").notna(),
            "pass_pre_count_filter": lambda s: s == True,
            "pass_ACE2bind_expr_filter": lambda s: s == True,
        }
    )

    drop_na_columns: List[str] = field(
        default_factory=lambda: [
            "name",
            "label",
            "pass_pre_count_filter",
            "pass_ACE2bind_expr_filter",
        ]
    )

    type_conversions: Dict[str, str] = field(default_factory=lambda: {"label": "float"})

    wt_sequence: str = (
        "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST"
    )

    label_columns: List[str] = field(default_factory=lambda: ["label"])

    primary_label_column: str = "label"

    input_is_zero_based: bool = False

    validate_mut_workers: int = 16

    process_workers: int = 16

    cache_validation_results: bool = False

    pipeline_name: str = "RBD_Antibody"

    def validate(self) -> None:
        """Validate RBD-antibody-specific configuration parameters.

        Raises
        ------
        ValueError
            If required label settings, column mappings, or WT sequence are missing.
        """
        super().validate()

        if not self.label_columns:
            raise ValueError("label_columns cannot be empty")

        if self.primary_label_column not in self.label_columns:
            raise ValueError(
                f"primary_label_column '{self.primary_label_column}' "
                f"must be in label_columns {self.label_columns}"
            )

        required_mappings = {
            "aa_substitutions",
            "score",
            "name",
            "variant_class",
            "pass_pre_count_filter",
            "pass_ACE2bind_expr_filter",
        }
        missing = required_mappings - set(self.column_mapping.keys())
        if missing:
            raise ValueError(f"Missing required column mappings: {missing}")

        if not self.wt_sequence:
            raise ValueError("wt_sequence cannot be empty")


def create_rbd_antibody_cleaner(
    dataset_or_path: Optional[Union[pd.DataFrame, str, Path, List[str]]] = None,
    config: Optional[Union[RBDAntibodyCleanerConfig, Dict[str, Any], str, Path]] = None,
) -> Pipeline:
    """Create the RBD antibody dataset cleaning pipeline.

    Parameters
    ----------
    dataset_or_path : Optional[Union[pd.DataFrame, str, Path, List[str]]], default=None
        Raw dataset DataFrame, dataset filepath, or a list of dataset filepaths.
    config : Optional[Union[RBDAntibodyCleanerConfig, Dict[str, Any], str, Path]], default=None
        Cleaner configuration object, partial configuration dictionary, JSON config path, or
        ``None`` to use the default configuration.

    Returns
    -------
    Pipeline
        Pipeline configured for RBD antibody data cleaning.

    Raises
    ------
    TypeError
        If ``dataset_or_path`` or ``config`` has an unsupported type.
    RuntimeError
        If pipeline creation fails.
    """
    if config is None:
        final_config = RBDAntibodyCleanerConfig()
    elif isinstance(config, RBDAntibodyCleanerConfig):
        final_config = config
    elif isinstance(config, dict):
        final_config = RBDAntibodyCleanerConfig().merge(config)
    elif isinstance(config, (str, Path)):
        final_config = RBDAntibodyCleanerConfig.from_json(config)
    else:
        raise TypeError("Invalid config type")

    logger.info(
        f"RBD antibody dataset will be cleaned with pipeline: {final_config.pipeline_name}"
    )
    logger.debug(f"Configuration:\n{final_config.get_summary()}")

    try:
        if isinstance(dataset_or_path, list):
            dataset_or_path = pd.concat(
                [pd.read_csv(p) for p in dataset_or_path],
                ignore_index=True,
            )
        elif dataset_or_path is not None and not isinstance(
            dataset_or_path, (pd.DataFrame, str, Path)
        ):
            raise TypeError(
                f"dataset_or_path must be pd.DataFrame, str, Path, list[str], or None, "
                f"got {type(dataset_or_path)}"
            )

        pipeline = create_pipeline(dataset_or_path, final_config.pipeline_name)

        # Standardize raw columns and explicit WT annotations before mutation
        # validation. Stop-codon rows are removed at this stage.
        pipeline = (
            pipeline.delayed_then(
                extract_and_rename_columns,
                column_mapping=final_config.column_mapping,
            )
            .delayed_then(
                convert_data_types,
                type_conversions=final_config.type_conversions,
            )
            .delayed_then(
                mark_wild_type_by_variant_class,
                mutation_column=final_config.column_mapping.get(
                    "aa_substitutions", "aa_substitutions"
                ),
                variant_class_column=final_config.column_mapping.get(
                    "variant_class", "variant_class"
                ),
                wild_type_value="wildtype",
                wt_identifier="WT",
            )
            .delayed_then(
                filter_and_clean_data,
                filters=final_config.filters,
                exclude_patterns={
                    final_config.column_mapping.get(
                        "aa_substitutions", "aa_substitutions"
                    ): [r"\*"]
                },
                drop_na_columns=final_config.drop_na_columns,
            )
            .delayed_then(
                add_column,
                dataset_name=final_config.wt_sequence,
                column_name="sequence",
            )
            .delayed_then(
                validate_mutations,
                mutation_column=final_config.column_mapping.get(
                    "aa_substitutions", "aa_substitutions"
                ),
                format_mutations=True,
                mutation_sep=" ",
                is_zero_based=final_config.input_is_zero_based,
                exclude_patterns=["WT"],
                cache_results=final_config.cache_validation_results,
                num_workers=final_config.validate_mut_workers,
            )
            .delayed_then(
                apply_mutations_preserving_wild_type,
                sequence_column="sequence",
                name_column=final_config.column_mapping.get("name", "name"),
                mutation_column=final_config.column_mapping.get(
                    "aa_substitutions", "aa_substitutions"
                ),
                mutation_sep=",",
                is_zero_based=True,
                num_workers=final_config.process_workers,
            )
            .delayed_then(
                average_labels_by_name,
                name_columns=(
                    final_config.column_mapping.get("name", "name"),
                    final_config.column_mapping.get(
                        "aa_substitutions", "aa_substitutions"
                    ),
                ),
                label_columns=final_config.primary_label_column,
            )
            .delayed_then(
                subtract_labels_by_wt,
                name_column=final_config.column_mapping.get("name", "name"),
                label_columns=final_config.primary_label_column,
                mutation_column=final_config.column_mapping.get(
                    "aa_substitutions", "aa_substitutions"
                ),
                wt_identifier="WT",
                in_place=True,
                drop_wt_row=False,
            )
            .delayed_then(
                convert_to_mutation_dataset_format,
                name_column=final_config.column_mapping.get("name", "name"),
                mutation_column=final_config.column_mapping.get(
                    "aa_substitutions", "aa_substitutions"
                ),
                mutated_sequence_column="mut_seq",
                label_column=final_config.primary_label_column,
                is_zero_based=True,
            )
        )

        if isinstance(dataset_or_path, (str, Path)):
            pipeline.add_delayed_step(read_dataset, 0, file_format="csv")
        elif dataset_or_path is not None and not isinstance(
            dataset_or_path, pd.DataFrame
        ):
            raise TypeError(
                f"dataset_or_path must be pd.DataFrame or str/Path, got {type(dataset_or_path)}"
            )

        return pipeline
    except Exception as e:
        logger.error(f"Error in creating RBD antibody cleaning pipeline: {str(e)}")
        raise RuntimeError(
            f"Error in creating RBD antibody cleaning pipeline: {str(e)}"
        )


def clean_rbd_antibody_dataset(
    pipeline: Pipeline,
) -> Tuple[Pipeline, MutationDataset]:
    """Execute the RBD antibody cleaning pipeline and build a MutationDataset.

    Parameters
    ----------
    pipeline : Pipeline
        RBD antibody cleaning pipeline created by :func:`create_rbd_antibody_cleaner`.

    Returns
    -------
    Tuple[Pipeline, MutationDataset]
        Executed pipeline and the cleaned MutationDataset.

    Raises
    ------
    RuntimeError
        If pipeline execution or dataset construction fails.
    """
    try:
        pipeline.execute()
        df, ref_seq = pipeline.data
        dataset = MutationDataset.from_dataframe(df, ref_seq)
        logger.info(
            f"Successfully cleaned RBD antibody dataset: "
            f"{len(df)} mutations from {len(ref_seq)} proteins"
        )
        return pipeline, dataset
    except Exception as e:
        logger.error(
            f"Error in running RBD antibody dataset cleaning pipeline: {str(e)}"
        )
        raise RuntimeError(
            f"Error in running RBD antibody dataset cleaning pipeline: {str(e)}"
        )
