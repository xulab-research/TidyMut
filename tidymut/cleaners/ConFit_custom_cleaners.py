# tidymut/cleaners/protein_gym_pipeline_func.py
from __future__ import annotations

import re
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING
from tqdm import tqdm
from functools import partial

from ..utils.cleaner_workers import validate_single_mutation_and_sequence
from ..core.sequence import ProteinSequence, DNASequence, RNASequence
from joblib import Parallel, delayed
if TYPE_CHECKING:
    from typing import (List, Tuple, Union)
from ..core.pipeline import pipeline_step, multiout_step

if TYPE_CHECKING:
    from typing import List, Tuple, Union

__all__ = ["read_ConFit_data"]


def __dir__() -> List[str]:
    return __all__

_MUT_RE = re.compile(r"^\s*([A-Z\*])(\d+)([A-Z\*])\s*$")

@pipeline_step
def read_ConFit_data(
    data_path: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_dir = Path(data_path)
    dfs = []
    tqdm.write(f"Reading ConFit data from {data_dir}")

    try:
        for protein_dir in data_dir.iterdir():
            if not protein_dir.is_dir():
                continue
            csv_path = protein_dir / "data.csv"
            fasta_path = protein_dir / "wt.fasta"
            if not (csv_path.exists() and fasta_path.exists()):
                continue
            lines = [l.strip() for l in fasta_path.read_text().splitlines() if l.strip()]
            header = lines[0] if lines and lines[0].startswith(">") else ""
            wt_seq = "".join(l for l in lines if not l.startswith(">"))

            m = re.search(r"/(\d+)-(\d+)", header)
            offset = (int(m.group(1)) - 1) if m else 0 
            df = pd.read_csv(csv_path)

            def shift_mutant(s: str) -> str:
                out = []
                for tok in str(s).split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    mm = _MUT_RE.match(tok)
                    out.append(tok if not mm else f"{mm.group(1)}{int(mm.group(2)) - offset}{mm.group(3)}")
                return ",".join(out)

            df["wt_seq"] = wt_seq
            df["mutant"] = df["mutant"].map(shift_mutant)
            df["name"] = f"ConFit_{protein_dir.name}"

            dfs.append(df)

        df_out = pd.concat(dfs, ignore_index=True)

        tqdm.write(f"Successfully read {len(dfs)} ConFit proteins.")

    except Exception as e:
        tqdm.write(f"Error reading ConFit data: {e}")

    return df_out

@multiout_step(main="success", failed="failed")
def validate_mutations_and_sequences(
    dataset: pd.DataFrame,
    wt_sequence_column: str = "wt_seq",
    name_column: str = "name",
    mutation_column: str = "mut_info",
    mut_sequence_column: str = "mut_seq",
    mutation_sep: str = ",",
    is_zero_based: bool = True,
    sequence_type: str = "protein",
    num_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    tqdm.write("Validating mutations and sequences...")

    # Validate required columns exist
    required_columns = [wt_sequence_column, name_column, mutation_column, mut_sequence_column]
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Select appropriate sequence class based on sequence_type
    sequence_type = sequence_type.lower()
    if sequence_type == "protein":
        SequenceClass = ProteinSequence
    elif sequence_type == "dna":
        SequenceClass = DNASequence
    elif sequence_type == "rna":
        SequenceClass = RNASequence
    else:
        raise ValueError(
            f"Unsupported sequence type: {sequence_type}. Must be 'protein', 'dna', or 'rna'"
        )

    _validate_single_mutation_and_sequence = partial(
        validate_single_mutation_and_sequence,
        dataset_columns=dataset.columns,
        wt_sequence_column=wt_sequence_column,
        name_column=name_column,
        mutation_column=mutation_column,
        mut_sequence_column=mut_sequence_column,
        mutation_sep=mutation_sep,
        is_zero_based=is_zero_based,
        sequence_class=SequenceClass,
    )

    # Parallel processing
    rows = dataset.itertuples(index=False, name=None)
    results = Parallel(n_jobs=num_workers, backend="loky")(
        delayed(_validate_single_mutation_and_sequence)(row)
        for row in tqdm(rows, total=len(dataset), desc="Validating mutations and sequences")
    )

    # Separate successful and failed results
    mutated_seqs, error_messages = map(list, zip(*results))

    result_dataset = dataset.copy()
    result_dataset["mut_seq"] = mutated_seqs
    result_dataset["error_message"] = error_messages

    success_mask = pd.notnull(result_dataset["mut_seq"])
    successful_dataset = result_dataset[success_mask].drop(columns=["error_message"])
    failed_dataset = result_dataset[~success_mask].drop(columns=["mut_seq"])

    tqdm.write(
        f"Mutation validation: {len(successful_dataset)} successful, {len(failed_dataset)} failed"
    )
    return successful_dataset, failed_dataset    
