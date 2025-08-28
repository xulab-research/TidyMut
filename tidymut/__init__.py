# tidymut/__init__.py
"""
TidyMut: A Python package for cleaning protein mutation data
"""

__author__ = "Yuxiang Tang"

from .core import (
    # Alphabet
    alphabet,
    # Codon
    codon,
    # Mutation
    mutation,
    # Sequence
    sequence,
    # Dataset
    MutationDataset,
    # Pipeline
    Pipeline,
    pipeline_step,
    multiout_step,
    create_pipeline,
)

from .cleaners import (
    cdna_proteolysis_cleaner,
    protein_gym_cleaner,
    human_domainome_cleaner,
)
from .utils.data_source import (
    list_datasets_with_built_in_cleaners,
    show_download_instructions,
)

from .utils.raw_data_downloader import (
    download,
    download_cdna_proteolysis_source_file,
    download_protein_gym_source_file,
    download_human_domainome_source_file,
)
