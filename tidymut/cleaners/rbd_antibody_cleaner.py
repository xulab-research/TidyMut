from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tidymut.cleaners.antitoxin_pard3_custom_cleaners import add_wild_type_sequence
from tidymut.cleaners.base_config import BaseCleanerConfig
from tidymut.cleaners.basic_cleaners import (
    average_labels_by_name,
    convert_data_types,
    convert_to_mutation_dataset_format,
    extract_and_rename_columns,
    filter_and_clean_data,
    read_dataset,
    validate_mutations,
)
from tidymut.cleaners.rbd_antibody_custom_cleaners import (
    apply_mutations_preserving_wild_type,
    build_mutation_column,
    capture_rbd_wt_score_table,
    drop_na_in_required_columns,
    normalize_aa_substitutions,
    prepare_rbd_standard_output,
    remove_stop_mutations,
    remove_wild_type_rows,
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
    Configuration class for the RBD antibody binding dataset cleaner.

    This cleaner is designed for tables like
    `SARS-CoV-2-RBD_MAP_AZ_Abs_scores.csv`, where:
    - `aa_substitutions` stores 1-based amino-acid mutations separated by spaces
    - `score` is the per-barcode binding score after negative logarithm transformation
    - identical `(name, aa_substitutions)` observations should be averaged

    Attributes
    ----------
    column_mapping : Dict[str, str]
        Mapping from raw antibody table columns to tidymut standard columns.
    filters : Dict[str, Callable]
        Filter conditions used to remove rows that fail score or QC checks.
    type_conversions : Dict[str, str]
        Data type conversion specifications for score and mutation counts.
    wt_sequence : str
        Reference RBD sequence shared by the antibody binding datasets.
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
            "n_aa_substitutions": "n_mutations",
            "score": "score",
            "name": "name",
            "pass_pre_count_filter": "pass_pre_count_filter",
            "pass_ACE2bind_expr_filter": "pass_ACE2bind_expr_filter",
        }
    )

    filters: Dict[str, Callable] = field(
        default_factory=lambda: {
            "score": lambda s: pd.to_numeric(s, errors="coerce").notna(),
            "pass_pre_count_filter": lambda s: s == True,
            "pass_ACE2bind_expr_filter": lambda s: s == True,
        }
    )

    type_conversions: Dict[str, str] = field(
        default_factory=lambda: {"score": "float", "n_mutations": "int"}
    )

    wt_sequence: str = (
        "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST"
    )

    label_columns: List[str] = field(default_factory=lambda: ["score"])
    primary_label_column: str = "score"
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
            "n_aa_substitutions",
            "score",
            "name",
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

        # Normalize raw columns and construct mutation annotations before validation.
        pipeline = (
            pipeline
            .delayed_then(
                extract_and_rename_columns,
                column_mapping=final_config.column_mapping,
            )
            .delayed_then(normalize_aa_substitutions)
            .delayed_then(
                filter_and_clean_data,
                filters=final_config.filters,
            )
            .delayed_then(
                convert_data_types,
                type_conversions=final_config.type_conversions,
            )
            .delayed_then(
                remove_stop_mutations,
                mutation_column="aa_substitutions",
            )
            .delayed_then(
                build_mutation_column,
                source_column="aa_substitutions",
                target_column="mut_info",
                mutation_count_column="n_mutations",
            )
            .delayed_then(
                add_wild_type_sequence,
                wt_sequence_column="sequence",
                wt_sequence=final_config.wt_sequence,
            )
        )

        # Validate mutations, aggregate scores, and build the final dataset outputs.
        pipeline = (
            pipeline
            .delayed_then(
                drop_na_in_required_columns,
                required_columns=[
                    "name",
                    "score",
                    "pass_pre_count_filter",
                    "pass_ACE2bind_expr_filter",
                ],
            )
            .delayed_then(
                validate_mutations,
                mutation_column="mut_info",
                format_mutations=True,
                mutation_sep=",",
                is_zero_based=final_config.input_is_zero_based,
                exclude_patterns=["WT"],
                cache_results=final_config.cache_validation_results,
                num_workers=final_config.validate_mut_workers,
            )
            .delayed_then(
                apply_mutations_preserving_wild_type,
                sequence_column="sequence",
                name_column="name",
                mutation_column="mut_info",
                mutation_sep=",",
                is_zero_based=True,
                num_workers=final_config.process_workers,
            )
            .delayed_then(
                average_labels_by_name,
                name_columns=("name", "mut_info"),
                label_columns=final_config.primary_label_column,
                remove_origin_columns=False,
            )
            .delayed_then(prepare_rbd_standard_output)
            .delayed_then(
                capture_rbd_wt_score_table,
                columns=["name", "mut_info", "score", "sequence", "mut_seq"],
            )
            .delayed_then(
                remove_wild_type_rows,
                mutation_column="mut_info",
            )
            .delayed_then(
                convert_to_mutation_dataset_format,
                name_column="name",
                mutation_column="mut_info",
                sequence_column="sequence",
                label_column=final_config.primary_label_column,
                is_zero_based=True,
            )
        )

        if isinstance(dataset_or_path, (str, Path)):
            pipeline.add_delayed_step(read_dataset, 0, file_format="csv")
        elif dataset_or_path is not None and not isinstance(dataset_or_path, pd.DataFrame):
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
        logger.error(f"Error in running RBD antibody dataset cleaning pipeline: {str(e)}")
        raise RuntimeError(
            f"Error in running RBD antibody dataset cleaning pipeline: {str(e)}"
        )
