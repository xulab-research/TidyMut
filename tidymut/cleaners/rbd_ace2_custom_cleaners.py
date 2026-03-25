from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

from tidymut.cleaners.basic_cleaners import apply_mutations_to_sequences
from tidymut.core.pipeline import multiout_step, pipeline_step

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple


__all__ = [
    "RBD_ACE2_REFERENCE_SEQUENCES",
    "RBD_ACE2_TARGET_NAME_ALIASES",
    "add_reference_sequences_by_target",
    "apply_mutations_preserving_wild_type",
    "capture_rbd_ace2_wt_score_table",
    "normalize_rbd_ace2_mutations",
    "normalize_rbd_ace2_target_names",
    "remove_stop_mutations",
    "remove_wild_type_rows",
]


def __dir__() -> List[str]:
    return __all__


# Alias table used to normalize target names across RBD ACE2 sub-datasets.
RBD_ACE2_TARGET_NAME_ALIASES = {
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
    "E484K": "Eta",
    "e484k": "Eta",
    "Wuhan_Hu_1_E484K": "Eta",
    "wuhan_hu_1_e484k": "Eta",
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


# Reference RBD sequences used as RBD ACE2 binding backgrounds.
RBD_ACE2_REFERENCE_SEQUENCES = {
    "Wuhan-Hu-1": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "Alpha": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "Beta": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "Eta": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "Delta": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGSKPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_BA1": "NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_BA2": "NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_BQ11": "NITNLCPFDEVFNATTFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSTVGGNYNYRYRLFRKSKLKPFERDISTEIYQAGNKPCNGVAGVNCYFPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_EG5": "NITNLCPFHEVFNATTFASVYAWNRKRISNCVADYSVIYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKPSGNYNYLYRLLRKSKLKPFERDISTEIYQAGNKPCNGVAGPNCYSPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_FLip": "NITNLCPFHEVFNATTFASVYAWNRKRISNCVADYSVIYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKPSGNYNYLYRFLRKSKLKPFERDISTEIYQAGNKPCNGVAGPNCYSPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
    "Omicron_XBB15": "NITNLCPFHEVFNATTFASVYAWNRKRISNCVADYSVIYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKPSGNYNYLYRLFRKSKLKPFERDISTEIYQAGNKPCNGVAGPNCYSPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST",
}


@pipeline_step
def normalize_rbd_ace2_target_names(
    dataset: pd.DataFrame,
    name_column: str = "name",
    name_aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Normalize raw RBD ACE2 target names to a shared reference naming scheme."""
    if name_column not in dataset.columns:
        raise ValueError(f"Target column '{name_column}' not found")

    aliases = name_aliases or RBD_ACE2_TARGET_NAME_ALIASES
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


@pipeline_step
def normalize_rbd_ace2_mutations(
    dataset: pd.DataFrame,
    mutation_column: str = "mut_info",
    mutation_count_column: Optional[str] = "n_mutations",
) -> pd.DataFrame:
    """Normalize RBD ACE2 mutation strings and mark rows with zero mutations as WT."""
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


@multiout_step(main="success", failed="failed")
def remove_stop_mutations(
    dataset: pd.DataFrame,
    mutation_column: str = "mut_info",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows containing stop mutations and return them as a failed artifact."""
    if mutation_column not in dataset.columns:
        raise ValueError(f"Mutation column '{mutation_column}' not found")

    removed = dataset[dataset[mutation_column].str.contains(r"\*", na=False)].copy()
    if not removed.empty:
        removed["error_message"] = "Removed stop-containing mutations"
    result = dataset[~dataset[mutation_column].str.contains(r"\*", na=False)].copy()
    tqdm.write(f"Removed stop-containing mutations: {len(dataset)} -> {len(result)} rows")
    return result, removed


@pipeline_step
def remove_wild_type_rows(
    dataset: pd.DataFrame,
    mutation_column: str = "mut_info",
) -> pd.DataFrame:
    """Drop WT rows before creating the final RBD ACE2 MutationDataset."""
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
    """Attach the correct reference sequence to each normalized RBD ACE2 target."""
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
    """Apply mutations while copying WT rows forward unchanged."""
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
def capture_rbd_ace2_wt_score_table(
    dataset: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Capture the WT-inclusive RBD ACE2 score table as a pipeline artifact."""
    wt_table = dataset.copy()
    if columns is not None:
        available_columns = [col for col in columns if col in wt_table.columns]
        wt_table = wt_table[available_columns].copy()
    return dataset, wt_table
