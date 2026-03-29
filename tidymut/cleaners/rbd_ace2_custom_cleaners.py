from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

from tidymut.cleaners.basic_cleaners import mark_wild_type_by_variant_class
from tidymut.core.pipeline import pipeline_step

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional


__all__ = [
    "RBD_ACE2_REFERENCE_SEQUENCES",
    "RBD_ACE2_TARGET_NAME_ALIASES",
    "add_reference_sequences_by_target",
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
def normalize_rbd_ace2_target_names(
    dataset: pd.DataFrame,
    name_column: str = "name",
    name_aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize raw RBD ACE2 target names to a shared reference naming scheme.

    This step maps source-specific target labels onto the canonical reference
    names used downstream for sequence lookup so the rest of the pipeline can
    operate on a single naming convention.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataframe containing the target/background name column.
    name_column : str, default="name"
        Column containing raw target/background labels to normalize.
    name_aliases : Optional[Dict[str, str]], default=None
        Mapping from raw source labels to canonical reference names. If ``None``,
        ``RBD_ACE2_TARGET_NAME_ALIASES`` is used.

    Returns
    -------
    pd.DataFrame
        A copied dataframe with normalized values in ``name_column``.

    Raises
    ------
    ValueError
        If ``name_column`` is not present in the input dataframe.

    Notes
    -----
    Unmatched target names are preserved as-is and reported with a warning so
    missing aliases can be added explicitly when new source labels appear.
    """
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
    """
    Attach the appropriate reference RBD sequence to each normalized target name.

    This step uses the normalized target/background label in ``name_column`` to
    look up the matching reference sequence from ``reference_sequences`` and
    stores the result in ``sequence_column``. Every target present in the input
    dataframe must have a corresponding reference entry.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataframe containing normalized target/background names.
    reference_sequences : Dict[str, str]
        Mapping from canonical target name to reference RBD sequence.
    name_column : str, default="name"
        Column containing the normalized target/background identifier.
    sequence_column : str, default="sequence"
        Output column that will store the matched reference sequence.

    Returns
    -------
    pd.DataFrame
        A copied dataframe with reference sequences added in ``sequence_column``.

    Raises
    ------
    ValueError
        If ``name_column`` is missing or if any target cannot be mapped to a
        reference sequence.
    """
    if name_column not in dataset.columns:
        raise ValueError(f"Target column '{name_column}' not found")

    result = dataset.copy()
    unknown_targets = sorted(set(result[name_column]) - set(reference_sequences))
    if unknown_targets:
        raise ValueError(f"Unknown targets without reference sequence: {unknown_targets}")

    result[sequence_column] = result[name_column].map(reference_sequences)
    return result



