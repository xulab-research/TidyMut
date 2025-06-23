# tidymut/cleaners/k50_cleaner_pipeline.py

import pandas as pd
from typing import Tuple

from .basic_cleaners import (
    extract_and_rename_columns,
    filter_and_clean_data,
    convert_data_types,
    validate_mutations,
    infer_wildtype_sequences,
    convert_to_mutation_dataset_format,
)
from ..core.dataset import MutationDataset
from ..core.pipeline import Pipeline, create_pipeline


def clean_k50_dataset(dataset: pd.DataFrame) -> Tuple[Pipeline, MutationDataset]:
    pipeline = create_pipeline(dataset, "k50_cleaner")
    clean_result = (
        pipeline.then(
            extract_and_rename_columns,
            column_mapping={
                "WT_name": "name",
                "aa_seq": "mut_seq",
                "mut_type": "mut_info",
                "ddG_ML": "ddG",
            },
        )
        .then(filter_and_clean_data, filters={"ddG": lambda x: x != "-"})
        .then(convert_data_types, type_conversions={"ddG": "float"})
        .then(
            validate_mutations,
            mutation_column="mut_info",
            mutation_sep="_",
            is_zero_based=False,
            num_workers=16,
        )
        .then(
            infer_wildtype_sequences,
            label_columns=["ddG"],
            handle_multiple_wt="error",
            is_zero_based=True,
            num_workers=16,
        )
        .then(
            convert_to_mutation_dataset_format,
            name_column="name",
            mutation_column="mut_info",
            mutated_sequence_column="mut_seq",
            score_column="ddG",
            is_zero_based=True,
        )
    )
    k50_dataset_df, k50_ref_seq = clean_result.data
    k50_dataset = MutationDataset.from_dataframe(k50_dataset_df, k50_ref_seq)
    return pipeline, k50_dataset
