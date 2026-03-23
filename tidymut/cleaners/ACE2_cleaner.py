from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

from tidymut.cleaners.base_config import BaseCleanerConfig
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
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)

ACE2_TARGET_NAME_ALIASES = {
    "Wuhan_Hu_1": "Wuhan-Hu-1",
    "Wuhan-Hu-1": "Wuhan-Hu-1",
    "wuhan_hu_1": "Wuhan-Hu-1",
    "wuhan-hu-1": "Wuhan-Hu-1",
    "wuhan hu 1": "Wuhan-Hu-1",
    "Alpha": "Alpha",
    "alpha": "Alpha",
    "Beta": "Beta",
    "beta": "Beta",
    "Delta": "Delta",
    "delta": "Delta",
    "Eta": "Eta",
    "eta": "Eta",
    "E484K": "E484K",
    "e484k": "E484K",
    "Wuhan_Hu_1_E484K": "E484K",
    "wuhan_hu_1_e484k": "E484K",
    "BA1": "Omicron_BA1",
    "BA.1": "Omicron_BA1",
    "Omicron_BA1": "Omicron_BA1",
    "Omicron BA1": "Omicron_BA1",
    "Omicron BA.1": "Omicron_BA1",
    "BA2": "Omicron_BA2",
    "BA.2": "Omicron_BA2",
    "Omicron_BA2": "Omicron_BA2",
    "Omicron BA2": "Omicron_BA2",
    "Omicron BA.2": "Omicron_BA2",
    "BA286": "Omicron_BA286",
    "BA.2.86": "Omicron_BA286",
    "Omicron_BA286": "Omicron_BA286",
    "Omicron BA286": "Omicron_BA286",
    "Omicron BA.2.86": "Omicron_BA286",
    "BQ11": "Omicron_BQ11",
    "BQ1.1": "Omicron_BQ11",
    "BQ.1.1": "Omicron_BQ11",
    "Omicron_BQ11": "Omicron_BQ11",
    "Omicron BQ11": "Omicron_BQ11",
    "Omicron BQ1.1": "Omicron_BQ11",
    "EG5": "Omicron_EG5",
    "EG.5": "Omicron_EG5",
    "Omicron_EG5": "Omicron_EG5",
    "Omicron EG5": "Omicron_EG5",
    "Omicron EG.5": "Omicron_EG5",
    "FLip": "Omicron_FLip",
    "flip": "Omicron_FLip",
    "Omicron_FLip": "Omicron_FLip",
    "Omicron FLip": "Omicron_FLip",
    "XBB15": "Omicron_XBB15",
    "XBB1.5": "Omicron_XBB15",
    "XBB.1.5": "Omicron_XBB15",
    "Omicron_XBB15": "Omicron_XBB15",
    "Omicron XBB15": "Omicron_XBB15",
    "Omicron XBB1.5": "Omicron_XBB15",
}

ACE2_REFERENCE_SEQUENCES = {
    "Wuhan-Hu-1": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "Alpha": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "Beta": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "Eta": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "E484K": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "Delta": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGSKPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_BA1": "NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_BA2": "NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_BA286": "NITNLCPFHEVFNATTFASVYAWNRKRISNCVADYSVIYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKPSGNYNYLYRFLRKSKLKPFERDISTEIYQAGNKPCNGVAGPNCYSPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_BQ11": "NITNLCPFDEVFNATTFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSTVGGNYNYRYRLFRKSKLKPFERDISTEIYQAGNKPCNGVAGVNCYFPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_EG5": "NITNLCPFHEVFNATTFASVYAWNRKRISNCVADYSVIYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKPSGNYNYLYRLLRKSKLKPFERDISTEIYQAGNKPCNGVAGPNCYSPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_FLip": "NITNLCPFHEVFNATTFASVYAWNRKRISNCVADYSVIYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKPSGNYNYLYRFLRKSKLKPFERDISTEIYQAGNKPCNGVAGPNCYSPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_XBB15": "NITNLCPFHEVFNATTFASVYAWNRKRISNCVADYSVIYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKPSGNYNYLYRLFRKSKLKPFERDISTEIYQAGNKPCNGVAGPNCYSPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
}

__all__ = [
    "ACE2CleanerConfig",
    "create_ACE2_cleaner",
    "clean_ACE2_dataset",
    "save_ACE2_outputs",
    "save_ACE2_wt_score_table",
]


def __dir__() -> List[str]:
    return __all__


@pipeline_step
def normalize_ace2_target_names(
    dataset: pd.DataFrame,
    name_column: str = "name",
    name_aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    if name_column not in dataset.columns:
        raise ValueError(f"Target column '{name_column}' not found")

    aliases = name_aliases or ACE2_TARGET_NAME_ALIASES
    result = dataset.copy()

    def normalize_name(value: Any) -> Any:
        if pd.isna(value):
            return value

        raw = str(value).strip()
        if raw in aliases:
            return aliases[raw]

        normalized_key = raw.replace("-", "_").replace(" ", "_")
        if normalized_key in aliases:
            return aliases[normalized_key]

        return raw

    result[name_column] = result[name_column].map(normalize_name)
    return result


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

    pd.DataFrame([{"error_message": reason, "count": len(removed_rows)}]).to_csv(
        debug_path / f"{debug_prefix}_removed_debug_reason_summary.csv",
        index=False,
    )


@pipeline_step
def normalize_ace2_mutations(
    dataset: pd.DataFrame,
    mutation_column: str = "mut_info",
    mutation_count_column: Optional[str] = "n_mutations",
) -> pd.DataFrame:
    if mutation_column not in dataset.columns:
        raise ValueError(f"Mutation column '{mutation_column}' not found")

    result = dataset.copy()

    def normalize_value(value: Any, mutation_count: Any) -> str:
        raw = "" if pd.isna(value) else str(value).strip()

        if mutation_count_column is not None and pd.notna(mutation_count):
            try:
                if int(mutation_count) == 0:
                    return "WT"
            except (TypeError, ValueError):
                pass

        if raw == "":
            return "WT"

        raw = raw.replace(";", ",")
        raw = re.sub(r"\s+", ",", raw)
        raw = re.sub(r",+", ",", raw).strip(",")
        return raw if raw else "WT"

    if mutation_count_column and mutation_count_column in result.columns:
        result[mutation_column] = [
            normalize_value(value, count)
            for value, count in zip(
                result[mutation_column].tolist(),
                result[mutation_count_column].tolist(),
            )
        ]
    else:
        result[mutation_column] = [
            normalize_value(value, None) for value in result[mutation_column].tolist()
        ]

    return result


@pipeline_step
def remove_stop_mutations(
    dataset: pd.DataFrame,
    mutation_column: str = "mut_info",
    debug_dir: Optional[Union[str, Path]] = None,
    debug_prefix: str = "ACE2_stop",
) -> pd.DataFrame:
    if mutation_column not in dataset.columns:
        raise ValueError(f"Mutation column '{mutation_column}' not found")

    removed = dataset[dataset[mutation_column].str.contains(r"\*", na=False)].copy()
    _write_removed_rows_debug(
        removed_rows=removed,
        debug_dir=debug_dir,
        debug_prefix=debug_prefix,
        reason="Removed stop-containing mutations",
    )
    result = dataset[~dataset[mutation_column].str.contains(r"\*", na=False)].copy()
    tqdm.write(f"Removed stop-containing mutations: {len(dataset)} -> {len(result)} rows")
    return result


@pipeline_step
def remove_wild_type_rows(
    dataset: pd.DataFrame,
    mutation_column: str = "mut_info",
) -> pd.DataFrame:
    if mutation_column not in dataset.columns:
        raise ValueError(f"Mutation column '{mutation_column}' not found")

    result = dataset[dataset[mutation_column].astype(str).str.upper() != "WT"].copy()
    tqdm.write(
        f"Removed WT rows before sequence application: {len(dataset)} -> {len(result)} rows"
    )
    return result


@pipeline_step
def add_reference_sequences_by_target(
    dataset: pd.DataFrame,
    reference_sequences: Dict[str, str],
    name_column: str = "name",
    sequence_column: str = "sequence",
) -> pd.DataFrame:
    if name_column not in dataset.columns:
        raise ValueError(f"Target column '{name_column}' not found")

    result = dataset.copy()
    unknown_targets = sorted(set(result[name_column]) - set(reference_sequences))
    if unknown_targets:
        raise ValueError(f"Unknown targets without reference sequence: {unknown_targets}")

    result[sequence_column] = result[name_column].map(reference_sequences)
    return result


@multiout_step(main="success", failed="failed")
def apply_mutations_preserving_wild_type(
    dataset: pd.DataFrame,
    sequence_column: str = "sequence",
    name_column: str = "name",
    mutation_column: str = "mut_info",
    mutation_sep: str = ",",
    is_zero_based: bool = True,
    sequence_type: str = "protein",
    num_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if mutation_column not in dataset.columns:
        raise ValueError(f"Mutation column '{mutation_column}' not found")

    wt_mask = dataset[mutation_column].astype(str).str.upper() == "WT"
    wt_rows = dataset[wt_mask].copy()
    mutant_rows = dataset[~wt_mask].copy()

    if not wt_rows.empty:
        wt_rows["mut_seq"] = wt_rows[sequence_column]

    if mutant_rows.empty:
        failed = pd.DataFrame(columns=list(dataset.columns) + ["error_message"])
        return wt_rows, failed

    mutant_success, mutant_failed = apply_mutations_to_sequences(
        mutant_rows,
        sequence_column=sequence_column,
        name_column=name_column,
        mutation_column=mutation_column,
        mutation_sep=mutation_sep,
        is_zero_based=is_zero_based,
        sequence_type=sequence_type,
        num_workers=num_workers,
    )

    success = pd.concat([mutant_success, wt_rows], axis=0).sort_index()
    return success, mutant_failed


@multiout_step(main="main", wt_table="wt_score_table")
def capture_ace2_wt_score_table(
    dataset: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wt_table = dataset.copy()
    if columns is not None:
        available_columns = [col for col in columns if col in wt_table.columns]
        wt_table = wt_table[available_columns].copy()
    return dataset, wt_table


@dataclass
class ACE2CleanerConfig(BaseCleanerConfig):
    """
    Configuration class for the ACE2 binding dataset cleaner.

    This cleaner is designed for RBD-ACE2 binding tables where:
    - `aa_substitutions` stores 1-based amino-acid mutations
    - `target` identifies the reference RBD background
    - `log10Ka` is the binding label to keep
    """

    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "target": "name",
            "aa_substitutions": "mut_info",
            "log10Ka": "label",
            "n_aa_substitutions": "n_mutations",
        }
    )

    filters: Dict[str, Callable] = field(
        default_factory=lambda: {
            "label": lambda s: pd.to_numeric(s, errors="coerce").notna(),
            "name": lambda s: s.isin(ACE2_REFERENCE_SEQUENCES),
        }
    )

    drop_na_columns: List[str] = field(
        default_factory=lambda: ["name", "label"]
    )

    type_conversions: Dict[str, str] = field(
        default_factory=lambda: {"label": "float", "n_mutations": "int"}
    )

    reference_sequences: Dict[str, str] = field(
        default_factory=lambda: ACE2_REFERENCE_SEQUENCES.copy()
    )
    target_name_aliases: Dict[str, str] = field(
        default_factory=lambda: ACE2_TARGET_NAME_ALIASES.copy()
    )
    keep_wild_type_scores: bool = False

    validate_mut_workers: int = 16
    process_workers: int = 16
    cache_validation_results: bool = False
    removed_stop_debug_dir: Optional[Union[str, Path]] = None
    removed_stop_debug_prefix: str = "ACE2_stop"

    label_columns: List[str] = field(default_factory=lambda: ["label"])
    primary_label_column: str = "label"

    pipeline_name: str = "ACE2_binding"

    def validate(self) -> None:
        super().validate()

        if not self.label_columns:
            raise ValueError("label_columns cannot be empty")

        if self.primary_label_column not in self.label_columns:
            raise ValueError(
                f"primary_label_column '{self.primary_label_column}' "
                f"must be in label_columns {self.label_columns}"
            )

        required_mappings = {"target", "aa_substitutions", "log10Ka"}
        missing = required_mappings - set(self.column_mapping.keys())
        if missing:
            raise ValueError(f"Missing required column mappings: {missing}")

        if not self.reference_sequences:
            raise ValueError("reference_sequences cannot be empty")


def create_ACE2_cleaner(
    dataset_or_path: Optional[Union[pd.DataFrame, str, Path, List[str]]] = None,
    config: Optional[Union[ACE2CleanerConfig, Dict[str, Any], str, Path]] = None,
) -> Pipeline:
    if config is None:
        final_config = ACE2CleanerConfig()
    elif isinstance(config, ACE2CleanerConfig):
        final_config = config
    elif isinstance(config, dict):
        final_config = ACE2CleanerConfig().merge(config)
    elif isinstance(config, (str, Path)):
        final_config = ACE2CleanerConfig.from_json(config)
    else:
        raise TypeError("Invalid config type")

    logger.info(
        f"ACE2 dataset will be cleaned with pipeline: {final_config.pipeline_name}"
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
            .delayed_then(
                normalize_ace2_target_names,
                name_column="name",
                name_aliases=final_config.target_name_aliases,
            )
            .delayed_then(
                convert_data_types,
                type_conversions=final_config.type_conversions,
            )
            .delayed_then(
                normalize_ace2_mutations,
                mutation_column="mut_info",
                mutation_count_column="n_mutations",
            )
            .delayed_then(
                remove_stop_mutations,
                mutation_column="mut_info",
                debug_dir=final_config.removed_stop_debug_dir,
                debug_prefix=final_config.removed_stop_debug_prefix,
            )
        )

        if final_config.keep_wild_type_scores:
            pipeline = (
                pipeline
                .delayed_then(
                    filter_and_clean_data,
                    filters=final_config.filters,
                    drop_na_columns=["name", final_config.primary_label_column],
                )
                .delayed_then(
                    validate_mutations,
                    mutation_column="mut_info",
                    format_mutations=True,
                    mutation_sep=",",
                    is_zero_based=False,
                    exclude_patterns=["WT"],
                    cache_results=final_config.cache_validation_results,
                    num_workers=final_config.validate_mut_workers,
                )
                .delayed_then(
                    average_labels_by_name,
                    name_columns=("name", "mut_info"),
                    label_columns=final_config.primary_label_column,
                )
                .delayed_then(
                    add_reference_sequences_by_target,
                    reference_sequences=final_config.reference_sequences,
                    name_column="name",
                    sequence_column="sequence",
                )
                .delayed_then(
                    apply_mutations_preserving_wild_type,
                    sequence_column="sequence",
                    name_column="name",
                    mutation_column="mut_info",
                    mutation_sep=",",
                    is_zero_based=True,
                    sequence_type="protein",
                    num_workers=final_config.process_workers,
                )
                .delayed_then(
                    capture_ace2_wt_score_table,
                    columns=[
                        "name",
                        "mut_info",
                        "n_mutations",
                        final_config.primary_label_column,
                        "sequence",
                        "mut_seq",
                    ],
                )
                .delayed_then(
                    convert_to_mutation_dataset_format,
                    name_column="name",
                    mutation_column="mut_info",
                    label_column=final_config.primary_label_column,
                    include_wild_type=True,
                    is_zero_based=True,
                )
            )
        else:
            pipeline = (
                pipeline
                .delayed_then(
                    filter_and_clean_data,
                    filters=final_config.filters,
                    drop_na_columns=final_config.drop_na_columns,
                )
                .delayed_then(
                    validate_mutations,
                    mutation_column="mut_info",
                    format_mutations=True,
                    mutation_sep=",",
                    is_zero_based=False,
                    exclude_patterns=["WT"],
                    cache_results=final_config.cache_validation_results,
                    num_workers=final_config.validate_mut_workers,
                )
                .delayed_then(
                    average_labels_by_name,
                    name_columns=("name", "mut_info"),
                    label_columns=final_config.primary_label_column,
                )
                .delayed_then(
                    remove_wild_type_rows,
                    mutation_column="mut_info",
                )
                .delayed_then(
                    add_reference_sequences_by_target,
                    reference_sequences=final_config.reference_sequences,
                    name_column="name",
                    sequence_column="sequence",
                )
                .delayed_then(
                    apply_mutations_to_sequences,
                    sequence_column="sequence",
                    name_column="name",
                    mutation_column="mut_info",
                    mutation_sep=",",
                    is_zero_based=True,
                    sequence_type="protein",
                    num_workers=final_config.process_workers,
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
        logger.error(f"Error in creating ACE2 cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in creating ACE2 cleaning pipeline: {str(e)}")


def clean_ACE2_dataset(
    pipeline: Pipeline,
) -> Tuple[Pipeline, MutationDataset]:
    try:
        pipeline.execute()

        df, ref_seq = pipeline.data
        dataset = MutationDataset.from_dataframe(df, ref_seq)

        logger.info(
            f"Successfully cleaned ACE2 dataset: "
            f"{len(df)} mutations from {len(ref_seq)} proteins"
        )

        return pipeline, dataset
    except Exception as e:
        logger.error(f"Error in running ACE2 dataset cleaning pipeline: {str(e)}")
        raise RuntimeError(f"Error in running ACE2 dataset cleaning pipeline: {str(e)}")


def save_ACE2_wt_score_table(
    pipeline: Pipeline,
    output_path: Union[str, Path],
) -> Dict[str, Path]:
    artifact_name = "capture_ace2_wt_score_table.wt_score_table"
    if artifact_name not in pipeline.artifacts:
        raise ValueError(
            "WT score table artifact not found. "
            "Create the cleaner with config={'keep_wild_type_scores': True} "
            "and execute the pipeline first."
        )

    wt_table = pipeline.get_artifact(artifact_name)
    if not isinstance(wt_table, pd.DataFrame):
        raise ValueError("WT score table artifact is missing or invalid")

    output_path = Path(output_path)
    saved_paths: Dict[str, Path] = {}
    output_path.mkdir(parents=True, exist_ok=True)
    for name, group in wt_table.groupby("name", sort=False, dropna=False):
        name_str = str(name)
        target_dir = output_path / name_str
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "data_with_wt_scores.csv"
        group.to_csv(target_file, index=False)
        saved_paths[name_str] = target_file

    logger.info(
        f"Saved ACE2 WT-inclusive score tables for {len(saved_paths)} references under {output_path}"
    )
    return saved_paths


def save_ACE2_outputs(
    pipeline: Pipeline,
    dataset: MutationDataset,
    output_dir: Union[str, Path],
) -> None:
    output_dir = Path(output_dir)
    dataset.save(output_dir)

    artifact_name = "capture_ace2_wt_score_table.wt_score_table"
    if artifact_name in pipeline.artifacts:
        save_ACE2_wt_score_table(pipeline, output_dir)
