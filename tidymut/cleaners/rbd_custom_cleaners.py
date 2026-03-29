from __future__ import annotations

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
    "mark_wild_type_by_variant_class",
    "normalize_rbd_ace2_target_names",
]


def __dir__() -> List[str]:
    return __all__


RBD_ACE2_TARGET_NAME_ALIASES = {
    "Wuhan_Hu_1": "Wuhan-Hu-1",
    "N501Y": "Alpha",
    "B1351": "Beta",
    "Delta": "Delta",
    "E484K": "Eta",
    "BA1": "Omicron_BA1",
    "BA2": "Omicron_BA2",
    "BQ11": "Omicron_BQ11",
    "EG5": "Omicron_EG5",
    "FLip": "Omicron_FLip",
    "XBB15": "Omicron_XBB15",
}


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


def _normalize_target_key(value: str) -> str:
    """Normalize a raw target label into a stable dictionary lookup key."""
    return value.strip().replace("-", "_").replace(" ", "_")


@pipeline_step
def mark_wild_type_by_variant_class(
    dataset: pd.DataFrame,
    mutation_column: str = "mut_info",
    variant_class_column: str = "variant_class",
    wild_type_value: str = "wildtype",
    wt_identifier: str = "WT",
) -> pd.DataFrame:
    """Rewrite mutation labels for rows explicitly annotated as wild type."""
    if mutation_column not in dataset.columns:
        raise ValueError(f"Mutation column '{mutation_column}' not found")
    if variant_class_column not in dataset.columns:
        raise ValueError(f"Variant-class column '{variant_class_column}' not found")

    result = dataset.copy()
    wt_mask = (
        result[variant_class_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        == wild_type_value.lower()
    )
    result[mutation_column] = (
        result[mutation_column].fillna("").astype(str).str.strip()
    )
    result.loc[wt_mask, mutation_column] = wt_identifier
    return result


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
    normalized_aliases = {_normalize_target_key(key): value for key, value in aliases.items()}
    result = dataset.copy()
    unmatched_names = set()

    def normalize_name(value: Any) -> Any:
        if pd.isna(value):
            return value

        raw = str(value).strip()
        if raw in aliases:
            return aliases[raw]

        normalized_key = _normalize_target_key(raw)
        if normalized_key in normalized_aliases:
            return normalized_aliases[normalized_key]

        unmatched_names.add(raw)
        return raw

    result[name_column] = result[name_column].map(normalize_name)
    if unmatched_names:
        tqdm.write(
            "Warning: Unrecognized RBD ACE2 target names: "
            f"{sorted(unmatched_names)}"
        )
    return result


@pipeline_step
def add_reference_sequences_by_target(
    dataset: pd.DataFrame,
    reference_sequences: Dict[str, str],
    name_column: str = "name",
    sequence_column: str = "sequence",
) -> pd.DataFrame:
    """Attach the appropriate reference RBD sequence to each normalized target name."""
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
    """Apply mutations while preserving explicit WT rows."""
    if mutation_column not in dataset.columns:
        raise ValueError(f"Mutation column '{mutation_column}' not found")

    wt_mask = dataset[mutation_column].astype(str).str.upper() == "WT"
    wt_rows = dataset[wt_mask].copy()
    mutant_rows = dataset[~wt_mask].copy()

    if not wt_rows.empty:
        wt_rows["mut_seq"] = wt_rows[sequence_column]

    if mutant_rows.empty:
        failed_dataset = pd.DataFrame(
            columns=list(dataset.columns) + ["error_message"]
        )
        return wt_rows, failed_dataset

    mutation_result = apply_mutations_to_sequences(
        mutant_rows,
        sequence_column=sequence_column,
        name_column=name_column,
        mutation_column=mutation_column,
        mutation_sep=mutation_sep,
        is_zero_based=is_zero_based,
        sequence_type=sequence_type,
        num_workers=num_workers,
    )
    mutant_success = mutation_result.main
    mutant_failed = mutation_result.side["failed"]

    successful_dataset = pd.concat([mutant_success, wt_rows], axis=0).sort_index()
    return successful_dataset, mutant_failed
