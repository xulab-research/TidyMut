from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tidymut.cleaners.base_config import BaseCleanerConfig
from tidymut.cleaners.antitoxin_pard3_custom_cleaners import add_wild_type_sequence
from tidymut.cleaners.basic_cleaners import (
    apply_mutations_to_sequences,
    average_labels_by_name,
    convert_data_types,
    convert_to_mutation_dataset_format,
    extract_and_rename_columns,
    filter_and_clean_data,
    read_dataset,
    validate_mutations,
)
from tidymut.core.dataset import MutationDataset
from tidymut.core.pipeline import (
    Pipeline,
    create_pipeline,
    multiout_step,
    pipeline_step,
)

if TYPE_CHECKING:
    from typing import Callable, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)

__all__ = [
    "RBDAntibodyCleanerConfig",
    "create_RBD_antibody_cleaner",
    "clean_RBD_antibody_dataset",
    "save_RBD_antibody_outputs",
    "save_RBD_antibody_wt_score_table",
]


def __dir__() -> List[str]:
    return __all__


def _write_removed_rows_debug(
    removed_rows: pd.DataFrame,
    debug_dir: Optional[Union[str, Path]],
    debug_prefix: str,
    reason: str,
) -> None:
    if debug_dir is None or removed_rows.empty:
        return

    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)

    removed_debug = removed_rows.copy()
    removed_debug["error_message"] = reason
    removed_debug.to_csv(
        debug_path / f"{debug_prefix}_removed_debug.csv",
        index=False,
    )

    pd.DataFrame(
        [{"error_message": reason, "count": len(removed_rows)}]
    ).to_csv(
        debug_path / f"{debug_prefix}_removed_debug_reason_summary.csv",
        index=False,
    )


@dataclass
class RBDAntibodyCleanerConfig(BaseCleanerConfig):
    """
    Configuration class for the RBD antibody escape dataset cleaner.

    This cleaner is designed for tables like
    `SARS-CoV-2-RBD_MAP_AZ_Abs_scores.csv`, where:
    - `aa_substitutions` stores 1-based amino-acid mutations separated by spaces
    - `score` is the per-barcode escape score
    - identical `(name, aa_substitutions)` observations should be averaged
    """

    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "aa_substitutions": "aa_substitutions",
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
        default_factory=lambda: {"score": "float"}
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
    removed_stop_debug_dir: Optional[Union[str, Path]] = None
    removed_stop_debug_prefix: str = "RBD_stop"
    keep_wild_type_scores: bool = False

    pipeline_name: str = "RBD_Antibody"

    def validate(self) -> None:
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
            "pass_pre_count_filter",
            "pass_ACE2bind_expr_filter",
        }
        missing = required_mappings - set(self.column_mapping.keys())
        if missing:
            raise ValueError(f"Missing required column mappings: {missing}")

        if not self.wt_sequence:
            raise ValueError("wt_sequence cannot be empty")


@pipeline_step
def normalize_aa_substitutions(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["aa_substitutions"] = result["aa_substitutions"].fillna("").astype(str)
    result["aa_substitutions"] = result["aa_substitutions"].replace(
        ["", " ", "NA", "N/A", "nan", "NaN", "<NA>"],
        "",
    )
    result["aa_substitutions"] = result["aa_substitutions"].str.strip()
    return result


@pipeline_step
def drop_na_in_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
) -> pd.DataFrame:
    return df.dropna(subset=required_columns).copy()


@pipeline_step
def remove_stop_mutations(
    df: pd.DataFrame,
    mutation_column: str = "aa_substitutions",
    debug_dir: Optional[Union[str, Path]] = None,
    debug_prefix: str = "RBD_stop",
) -> pd.DataFrame:
    removed = df[df[mutation_column].str.contains(r"\*", na=False)].copy()
    _write_removed_rows_debug(
        removed_rows=removed,
        debug_dir=debug_dir,
        debug_prefix=debug_prefix,
        reason="Removed stop-containing mutations",
    )
    result = df[~df[mutation_column].str.contains(r"\*", na=False)].copy()
    print(
        f"Removed stop-containing mutations: {len(df)} -> {len(result)} "
        f"(dropped {len(removed)})"
    )
    return result


@pipeline_step
def remove_empty_mutations(
    df: pd.DataFrame,
    mutation_column: str = "aa_substitutions",
) -> pd.DataFrame:
    return df[df[mutation_column] != ""].copy()


@pipeline_step
def build_mutation_column(
    df: pd.DataFrame,
    source_column: str = "aa_substitutions",
    target_column: str = "mut_info",
) -> pd.DataFrame:
    result = df.copy()
    normalized = (
        result[source_column]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", ",", regex=True)
    )
    result[target_column] = normalized.where(normalized != "", "WT")
    return result


@pipeline_step
def prepare_rbd_standard_output(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "score_mean_by_name" in result.columns:
        result["score"] = result["score_mean_by_name"]

    output_columns = ["name", "mut_info", "score", "sequence", "mut_seq"]
    return result[output_columns].drop_duplicates(subset=["name", "mut_info"]).copy()


@pipeline_step
def remove_wild_type_rows(
    df: pd.DataFrame,
    mutation_column: str = "mut_info",
) -> pd.DataFrame:
    return df[df[mutation_column].astype(str).str.upper() != "WT"].copy()


@multiout_step(main="success", failed="failed")
def apply_mutations_preserving_wild_type(
    df: pd.DataFrame,
    sequence_column: str = "sequence",
    name_column: str = "name",
    mutation_column: str = "mut_info",
    mutation_sep: str = ",",
    is_zero_based: bool = True,
    num_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wt_mask = df[mutation_column].astype(str).str.upper() == "WT"
    wt_rows = df[wt_mask].copy()
    mutant_rows = df[~wt_mask].copy()

    if not wt_rows.empty:
        wt_rows["mut_seq"] = wt_rows[sequence_column]

    if mutant_rows.empty:
        failed = pd.DataFrame(columns=list(df.columns) + ["error_message"])
        return wt_rows, failed

    mutant_success, mutant_failed = apply_mutations_to_sequences(
        mutant_rows,
        sequence_column=sequence_column,
        name_column=name_column,
        mutation_column=mutation_column,
        mutation_sep=mutation_sep,
        is_zero_based=is_zero_based,
        num_workers=num_workers,
    )
    success = pd.concat([mutant_success, wt_rows], axis=0).sort_index()
    return success, mutant_failed


@multiout_step(main="main", wt_table="wt_score_table")
def capture_rbd_wt_score_table(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wt_table = df.copy()
    if columns is not None:
        available_columns = [col for col in columns if col in wt_table.columns]
        wt_table = wt_table[available_columns].copy()
    return df, wt_table


def create_RBD_antibody_cleaner(
    dataset_or_path: Optional[Union[pd.DataFrame, str, Path, List[str]]] = None,
    config: Optional[Union[RBDAntibodyCleanerConfig, Dict, str, Path]] = None,
) -> Pipeline:
    """Create RBD antibody dataset cleaning pipeline."""
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
                debug_dir=final_config.removed_stop_debug_dir,
                debug_prefix=final_config.removed_stop_debug_prefix,
            )
            .delayed_then(
                build_mutation_column,
                source_column="aa_substitutions",
                target_column="mut_info",
            )
            .delayed_then(
                add_wild_type_sequence,
                wt_sequence_column="sequence",
                wt_sequence=final_config.wt_sequence,
            )
        )

        if final_config.keep_wild_type_scores:
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
        else:
            pipeline = (
                pipeline
                .delayed_then(
                    drop_na_in_required_columns,
                    required_columns=[
                        "name",
                        "score",
                        "aa_substitutions",
                        "pass_pre_count_filter",
                        "pass_ACE2bind_expr_filter",
                    ],
                )
                .delayed_then(remove_empty_mutations)
                .delayed_then(
                    validate_mutations,
                    mutation_column="mut_info",
                    format_mutations=True,
                    mutation_sep=",",
                    is_zero_based=final_config.input_is_zero_based,
                    cache_results=final_config.cache_validation_results,
                    num_workers=final_config.validate_mut_workers,
                )
                .delayed_then(
                    apply_mutations_to_sequences,
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


def clean_RBD_antibody_dataset(
    pipeline: Pipeline,
) -> Tuple[Pipeline, MutationDataset]:
    """Clean RBD antibody dataset using configurable pipeline."""
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


def save_RBD_antibody_wt_score_table(
    pipeline: Pipeline,
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    artifact_name = "capture_rbd_wt_score_table.wt_score_table"
    if artifact_name not in pipeline.artifacts:
        raise ValueError(
            "WT score table artifact not found. "
            "Create the cleaner with config={'keep_wild_type_scores': True} "
            "and execute the pipeline first."
        )

    wt_table = pipeline.get_artifact(artifact_name)
    if not isinstance(wt_table, pd.DataFrame):
        raise ValueError("WT score table artifact is missing or invalid")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, Path] = {}
    for name, group in wt_table.groupby("name", sort=False, dropna=False):
        name_str = str(name)
        target_dir = output_dir / name_str
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "data_with_wt_scores.csv"
        group.to_csv(target_file, index=False)
        saved_paths[name_str] = target_file
    return saved_paths


def save_RBD_antibody_outputs(
    pipeline: Pipeline,
    dataset: MutationDataset,
    output_dir: Union[str, Path],
) -> None:
    output_dir = Path(output_dir)
    dataset.save(output_dir)
    if "capture_rbd_wt_score_table.wt_score_table" in pipeline.artifacts:
        save_RBD_antibody_wt_score_table(pipeline, output_dir)
