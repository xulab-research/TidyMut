import tqdm
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..core.mutation import MutationSet
from ..core.pipeline import pipeline_step

__all__ = ["compute_mutations"]

@pipeline_step
def compute_mutations(
    dataset: pd.DataFrame,
    name_column: str = "name",
    WT_column: str = "WT",
    mut_seq: str = "mut_seq"
) -> pd.DataFrame:
    """compute the mutations by the wt_seq and mut_seq and generate mutation column

    Parameters
    ----------
    dataset : pandas.DataFrame
    name_column : str
        Grouping key column.
    WT_column : str
        column used to check whether wt or mut
    mut_seq : str
        Column containing the amino-acid sequence for that row (treated as the
        mutated sequence for mutants and the WT sequence for the WT row).
    """
    def get_mut_info(group):
        # get wt sequence
        wt_rows = group[group[WT_column] == True]
        if len(wt_rows) == 0:
            return pd.Series([""] * len(group), index=group.index)
        wt_seq = wt_rows[mut_seq].values[0] # means wt_seq
        wt_array = np.array(list(wt_seq))

        # convert sequence to character matrix
        aa_array = np.array([list(seq) for seq in group[mut_seq]])
        diff_mask = aa_array != wt_array

        # generate mutation info for each sequence
        name = group[name_column].iloc[0]
        desc = f"Processing name_column {name}"
        mut_info_list = []
        for i, row in enumerate(tqdm(aa_array, desc=desc)):
            positions = np.where(diff_mask[i])[0]
            if len(positions) == 0:  # WT
                mut_info_list.append("WT")
                continue
            muts = [f"{wt_array[pos]}{pos}{row[pos]}" for pos in positions]
            mut_str = ",".join(muts)
            mut_info_list.append(
                str(MutationSet.from_string(mut_str, is_zero_based=True)) if mut_str else ""
            )
        return pd.Series(mut_info_list, index=group.index)

    dataset["mut_info"] = dataset.groupby(
        [name_column], group_keys=False
    ).apply(get_mut_info)

    dataset = dataset.drop(columns=WT_column)
    return dataset

