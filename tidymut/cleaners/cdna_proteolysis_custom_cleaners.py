# cleaners/cdna_proteolysis_custom_cleaners.py
from __future__ import annotations

import pandas as pd
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import TYPE_CHECKING

from ..core.mutation import MutationSet
from ..core.pipeline import multiout_step
from ..core.sequence import ProteinSequence
from ..utils.mutation_converter import invert_mutation_set

if TYPE_CHECKING:
    from typing import Any, Dict, List, Sequence, Tuple, Union


__all__ = [
    "validate_wt_sequence",
    "subtract_labels_by_wt",
]


def __dir__() -> List[str]:
    return __all__


@multiout_step(main="success", failed="failed")
def validate_wt_sequence(
    dataset: pd.DataFrame,
    name_column: str,
    mutation_column: str,
    sequence_column: str,
    wt_identifier: str = "wt",
    num_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate wild-type (WT) sequences per protein/name group by checking
    consistency between the *explicit* WT row and the WT sequence *inferred*
    from each mutant row.

    This step groups the dataset by ``name_column`` and, for each group:
      1) Ensures a WT row is present (i.e., ``mutation_column == wt_identifier``).
      2) Treats each row's sequence as the mutated sequence and parses the
         mutation string using ``MutationSet.from_string(sep="_", is_zero_based=True)``.
      3) Inverts the mutation set (``invert_mutation_set``) and applies it to the
         per-row sequence to infer that row's WT sequence.
      4) Fails the group if (a) multiple distinct inferred WT sequences are found,
         or (b) the single inferred WT sequence does not match the explicit WT
         sequence from the WT row. Otherwise the group is marked as success.

    Parallel execution is performed with ``joblib.Parallel(backend="loky")`` and
    a progress bar via ``tqdm``. If parallel execution fails, a sequential
    fallback is used. Diagnostic warnings are printed with ``tqdm.write``.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataframe containing at least the name, mutation, and sequence columns.
    name_column : str
        Column that identifies a protein/name group for validation.
    mutation_column : str
        Column containing mutation annotations. Its format must be parsable by
        ``MutationSet.from_string(sep="_", is_zero_based=True)`` (the project’s
        standard mutation string convention).
    sequence_column : str
        Column containing the amino-acid sequence for that row (treated as the
        mutated sequence for mutants and the WT sequence for the WT row).
    wt_identifier : str, default "wt"
        The exact token used in ``mutation_column`` to mark the explicit WT row.
    num_workers : int, default 4
        Number of worker processes for parallel execution.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - successful_df:
            Concatenation of **original rows** for groups that passed validation.
            No new rows are synthesized; both mutant and WT rows from successful
            groups are returned as-is. May be empty if no group passes.
        - failed_df:
            Rows summarizing groups that failed validation. Each failed group is
            represented by at least one row (derived from the group's first row)
            with an additional column ``"error_message"`` describing the reason,
            e.g., missing WT row, multiple inferred WT sequences, or mismatch
            between inferred and explicit WT sequences. May be empty.

    Examples
    --------
    >>> # Minimal schema example (mutation format must match your project's parser):
    >>> df = pd.DataFrame({
    ...     "protein": ["P1", "P1", "P1", "P2", "P2"],
    ...     "mut":     ["wt", "A0G", "L4F", "wt", "K3R"],
    ...     "seq":     ["ACDELG", "GCDELG", "ACDEFG", "MNPKQ", "MNPRQ"],
    ... })
    >>> successful, failed = validate_wt_sequence(
    ...     df, name_column="protein", mutation_column="mut", sequence_column="seq",
    ...     wt_identifier="wt", num_workers=2
    ... )
    >>> successful
      protein  mut     seq
    0      P1  A0G  GCDELG
    1      P1  L4F  ACDEFG
    2      P1   wt  ACDELG
    3      P2  K3R   MNPRQ
    4      P2   wt   MNPKQ

    See Also
    --------
    validate_wt_sequence_grouped : The per-group worker that validates a single name/protein group.
    """
    _process_protein_group = partial(
        validate_wt_sequence_grouped,
        name_column=name_column,
        mutation_column=mutation_column,
        sequence_column=sequence_column,
        wt_identifier=wt_identifier,
    )

    # Group by protein and process in parallel
    grouped = list(dataset.groupby(name_column, sort=False))

    try:
        results = Parallel(n_jobs=num_workers, backend="loky")(
            delayed(_process_protein_group)(group_data)
            for group_data in tqdm(grouped, desc="Processing proteins")
        )
    except Exception as e:
        tqdm.write(
            f"Warning: Parallel processing failed, falling back to sequential: {e}"
        )
        # Fallback to sequential processing
        results = []
        for group_data in tqdm(grouped, desc="Processing proteins (sequential)"):
            try:
                result = _process_protein_group(group_data)
                results.append(result)
            except Exception as group_e:
                # Create error entry for this specific group
                protein_name = group_data[0]
                error_row = {
                    name_column: str(protein_name),
                    "error_message": f"Sequential processing error: {type(group_e).__name__}: {str(group_e)}",
                }
                results.append(([error_row], "failed"))

    # Filter out None results and validate structure
    valid_results = []
    invalid_count = 0

    for i, result in enumerate(results):
        if result is None:
            invalid_count += 1
            tqdm.write(f"Warning: Result {i} is None, skipping")
            continue

        if not isinstance(result, tuple) or len(result) != 2:
            invalid_count += 1
            tqdm.write(f"Warning: Result {i} has invalid format, skipping: {result}")
            continue

        rows_list, category = result
        if category not in ("success", "failed"):
            invalid_count += 1
            tqdm.write(
                f"Warning: Result {i} has invalid category '{category}', skipping"
            )
            continue

        if not isinstance(rows_list, list):
            invalid_count += 1
            tqdm.write(f"Warning: Result {i} has invalid rows format, skipping")
            continue

        valid_results.append(result)

    if invalid_count > 0:
        tqdm.write(f"Warning: {invalid_count} invalid results were skipped")

    # Collect all rows
    successful_rows = []
    failed_rows = []

    for rows_list, category in valid_results:
        if category == "success":
            successful_rows.extend(rows_list)
        else:
            failed_rows.extend(rows_list)

    # Convert to DataFrame format
    successful_df = pd.DataFrame(successful_rows) if successful_rows else pd.DataFrame()
    failed_df = pd.DataFrame(failed_rows) if failed_rows else pd.DataFrame()

    tqdm.write(f"Success: {len(successful_df)}, Failed: {len(failed_df)}")

    return successful_df, failed_df


@multiout_step(main="successful", failed="failed")
def subtract_labels_by_wt(
    dataset: pd.DataFrame,
    name_column: str,
    label_columns: Union[str, Sequence[str]],
    mutation_column: str,
    wt_identifier: str = "wt",
    suffix: str = "_minus_wt",
    in_place: bool = False,
    drop_wt_row: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each group defined by ``name_column``, subtract the group's WT label values
    (where ``mutation_column == wt_identifier``) from all rows' label values.

    Returns two dataframes via the multi-output step:
    - **successful**: rows of groups where subtraction succeeded
      (columns overwritten if ``in_place=True``; otherwise new columns ``<label><suffix>`` are added).
      If ``drop_wt_row=True``, WT rows are removed in this output.
    - **failed**: one (or more) diagnostic rows per failed group, including an
      ``error_message`` column describing the reason.

    Failure conditions (per group)
    ------------------------------
    - No WT row found
    - Multiple WT rows found
    - Any WT label value is NaN
    - Arithmetic/type error during subtraction

    Parameters
    ----------
    dataset : pandas.DataFrame
    name_column : str
        Grouping key column.
    label_columns : Union[str, Sequence[str]]
        Numeric label column(s) to subtract by WT.
    mutation_column : str
        Column that identifies the WT row with ``wt_identifier``.
    wt_identifier : str, default "wt"
        Token used to mark the WT row.
    suffix : str, default "_minus_wt"
        Suffix for new columns (ignored if ``in_place=True``).
    in_place : bool, default False
        If True, overwrite original label columns with the deltas.
        If False, add new columns named ``<label><suffix>``.
    drop_wt_row : bool, default False
        If True, drop WT rows from the successful output.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "name": ["A","A","B","B","B","C","C"],
    ...     "mut" : ["wt","A0G","wt","K2R","L3F","S1N","T4A"],
    ...     "y1"  : [10.0,  12.0, 20.0,  22.0,  19.0,  8.0,   6.0],
    ...     "y2"  : [ 5.0,   4.0, 15.0,  10.0,  18.0,  3.0,   2.0],
    ... })
    >>> successful, failed = subtract_labels_by_wt(
    ...     dataset=df,
    ...     name_column="name",
    ...     label_columns=["y1", "y2"],
    ...     mutation_column="mut",
    ...     wt_identifier="wt",
    ...     in_place=True,
    ...     drop_wt_row=True,
    ... )
    >>> successful
      name  mut   y1   y2
    0    A  A0G  2.0 -1.0
    1    B  K2R  2.0 -5.0
    2    B  L3F -1.0  3.0
    >>> failed["failed"]
      name  mut   y1   y2                        error_message
    0    C  S1N  8.0  3.0  No WT row (mutation_column == 'wt')
    """
    # Basic column presence checks (dataset-level)
    required = {name_column, mutation_column}
    missing_req = [c for c in required if c not in dataset.columns]
    if missing_req:
        raise KeyError(f"required columns missing: {missing_req}")

    label_cols = (
        [label_columns] if isinstance(label_columns, str) else list(label_columns)
    )
    if not label_cols:
        raise KeyError("label_columns must be a non-empty str or sequence")
    missing_labels = [c for c in label_cols if c not in dataset.columns]
    if missing_labels:
        raise KeyError(f"label_columns not found: {missing_labels}")

    success_rows = []
    failed_rows = []

    # Iterate groups preserving order
    for _, grp in tqdm(
        dataset.groupby(name_column, sort=False), desc="Processing groups"
    ):
        try:
            wt_rows = grp[grp[mutation_column] == wt_identifier]
            if wt_rows.empty:
                err = grp.iloc[0].to_dict()
                err["error_message"] = (
                    f"No WT row (mutation_column == {wt_identifier!r})"
                )
                failed_rows.append(err)
                continue
            if len(wt_rows) > 1:
                err = wt_rows.iloc[0].to_dict()
                err["error_message"] = "Multiple WT rows found"
                failed_rows.append(err)
                continue

            # Extract WT label values
            wt_vals = wt_rows.iloc[0][label_cols]
            if wt_vals.isna().any():
                err = wt_rows.iloc[0].to_dict()
                err["error_message"] = (
                    f"WT label has NaN in columns: {list(wt_vals[wt_vals.isna()].index)}"
                )
                failed_rows.append(err)
                continue

            # Perform subtraction (vectorized)
            deltas = grp[label_cols].sub(wt_vals, axis=1)

            # Build output block for this group
            block = grp.copy()
            if in_place:
                block.loc[:, label_cols] = deltas.values
            else:
                new_names = {c: f"{c}{suffix}" for c in label_cols}
                block.loc[:, list(new_names.values())] = deltas.rename(
                    columns=new_names
                ).values

            if drop_wt_row:
                block = block[block[mutation_column] != wt_identifier]

            # Append rows to success list
            success_rows.extend(block.to_dict(orient="records"))

        except Exception as e:
            row = grp.iloc[0].to_dict()
            row["error_message"] = f"{type(e).__name__}: {e}"
            failed_rows.append(row)

    successful_df = pd.DataFrame(success_rows) if success_rows else pd.DataFrame()
    failed_df = pd.DataFrame(failed_rows) if failed_rows else pd.DataFrame()
    tqdm.write(f"Success: {len(successful_df)}, Failed: {len(failed_df)}")
    return successful_df, failed_df


def validate_wt_sequence_grouped(
    group_data: Tuple[Any, pd.DataFrame],
    name_column: str,
    mutation_column: str,
    sequence_column: str,
    wt_identifier: str = "wt",
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Validate one protein/name group by comparing the explicit WT row with
    the WT sequence inferred from each mutant row.
    """
    protein_name, original_group = group_data

    try:
        # Check WT sequence is valid
        wt_rows = original_group[original_group[mutation_column] == wt_identifier]
        if wt_rows.empty:
            error_row = original_group.iloc[0].to_dict()
            error_row["error_message"] = "WT row not found"
            return [error_row], "failed"
        if len(wt_rows) > 1:
            error_row = wt_rows.iloc[0].to_dict()
            error_row["error_message"] = "Multiple explicit WT rows"
            return [error_row], "failed"
        # Extract original WT sequence
        explicit_wt_seq = str(wt_rows.iloc[0][sequence_column]).strip()
        wt_row_dict = wt_rows.iloc[0].to_dict()

        mutants = original_group[original_group[mutation_column] != wt_identifier]
        if mutants.empty:
            # Return success if no mutants
            return [wt_row_dict], "success"

        # Infer wild-type sequences
        inferred_wt_seqs = set()
        result_rows = []

        # First, add all original rows to result
        for _, row in mutants.iterrows():
            result_rows.append(row.to_dict())

        # Then, infer WT sequences
        for _, row in mutants.iterrows():
            mut_info = row[mutation_column]
            mut_seq = row[sequence_column]

            # Parse mutation and create sequence
            mutation_set = MutationSet.from_string(
                mut_info, sep=",", is_zero_based=True
            )
            sequence = ProteinSequence(str(mut_seq).strip())

            # Infer wild-type sequence by applying inverted mutations
            inverted_mutation_set = invert_mutation_set(mutation_set)
            wt_seq = sequence.apply_mutation(inverted_mutation_set)
            inferred_wt_seqs.add(str(wt_seq))

        if len(inferred_wt_seqs) > 1:
            # Add error information to the first row
            error_row = mutants.iloc[0].to_dict()
            error_row["error_message"] = (
                f"Multiple wildtype sequences inferred for {protein_name}: {len(inferred_wt_seqs)}"
            )
            return [error_row], "failed"

        # Compare inferred WT sequences with original WT sequences
        if explicit_wt_seq != inferred_wt_seqs.pop():
            error_row = mutants.iloc[0].to_dict()
            error_row["error_message"] = (
                f"Inferred WT sequence differs from original WT sequence: {wt_seq} vs. {inferred_wt_seqs.pop()}"
            )
            return [error_row], "failed"
        else:
            result_rows.append(wt_row_dict)
            return result_rows, "success"

    except Exception as e:
        # Save error information in first row
        error_row = (
            mutants.iloc[0].to_dict()
            if len(mutants) > 0
            else {name_column: str(protein_name)}
        )
        error_row["error_message"] = f"{type(e).__name__}: {str(e)}"
        return [error_row], "failed"
