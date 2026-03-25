from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tidymut.cleaners.basic_cleaners import apply_mutations_to_sequences
from tidymut.core.pipeline import multiout_step, pipeline_step

if TYPE_CHECKING:
    from typing import List, Optional, Tuple


__all__ = [
    "apply_mutations_preserving_wild_type",
    "build_mutation_column",
    "capture_rbd_wt_score_table",
    "drop_na_in_required_columns",
    "normalize_aa_substitutions",
    "prepare_rbd_standard_output",
    "remove_empty_mutations",
    "remove_stop_mutations",
    "remove_wild_type_rows",
]


def __dir__() -> List[str]:
    return __all__


@pipeline_step
def normalize_aa_substitutions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw antibody mutation strings and standardize empty values."""
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
    """Drop rows missing required columns before mutation processing."""
    return df.dropna(subset=required_columns).copy()


@multiout_step(main="success", failed="failed")
def remove_stop_mutations(
    df: pd.DataFrame,
    mutation_column: str = "aa_substitutions",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows containing stop mutations and return them as a failed artifact."""
    removed = df[df[mutation_column].str.contains(r"\*", na=False)].copy()
    if not removed.empty:
        removed["error_message"] = "Removed stop-containing mutations"
    result = df[~df[mutation_column].str.contains(r"\*", na=False)].copy()
    return result, removed


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
    mutation_count_column: str = "n_mutations",
) -> pd.DataFrame:
    """Build the tidymut mutation column and mark zero-mutation rows as WT."""
    result = df.copy()
    normalized = (
        result[source_column]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", ",", regex=True)
    )
    if mutation_count_column in result.columns:
        result[target_column] = normalized.where(result[mutation_count_column] > 0, "WT")
    else:
        result[target_column] = normalized.where(normalized != "", "WT")
    return result


@pipeline_step
def prepare_rbd_standard_output(df: pd.DataFrame) -> pd.DataFrame:
    """Select the standard output columns and keep one row per mutation entry."""
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
    """Drop WT rows before creating the final RBD antibody MutationDataset."""
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
    """Apply mutations while copying WT rows forward unchanged."""
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
    """Capture the WT-inclusive antibody score table as a pipeline artifact."""
    wt_table = df.copy()
    if columns is not None:
        available_columns = [col for col in columns if col in wt_table.columns]
        wt_table = wt_table[available_columns].copy()
    return df, wt_table
