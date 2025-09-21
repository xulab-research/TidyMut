# tidymut/utils/data_source.py

DATASETS = {
    "cDNAProteolysis": {
        "name": "Mega-scale experimental analysis of protein folding stability in biology and design",
        "official_url": "https://zenodo.org/records/7992926",
        "files": [
            "'Tsuboyama2023_Dataset2_Dataset3_20230416.csv' in 'Processed_K50_dG_datasets.zip'"
        ],
        "huggingface_repos": [
            "datasets/xulab-research/TidyMut/resolve/main/cDNA_proteolysis/Tsuboyama2023_Dataset2_Dataset3_20230416.csv?download=true"
        ],
        "file_name": ["Tsuboyama2023_Dataset2_Dataset3_20230416.csv"],
    },
    "ProteinGym": {
        "name": "ProteinGym",
        "official_url": "https://proteingym.org/download",
        "files": ["DMS_ProteinGym_substitutions.zip"],
        "huggingface_repos": [
            "datasets/xulab-research/TidyMut/resolve/main/ProteinGym_DMS_substitutions/DMS_ProteinGym_substitutions.zip?download=true"
        ],
        "file_name": ["ProteinGym_DMS_substitutions.zip"],
    },
    "HumanDomainome": {
        "name": "Site-saturation mutagenesis of 500 human protein domains",
        "official_url": "https://www.nature.com/articles/s41586-024-08370-4",
        "files": [
            "SupplementaryTable2.txt",
            "SupplementaryTable4.txt",
            "wild_type.fasta",
        ],
        "huggingface_repos": [
            "datasets/xulab-research/TidyMut/resolve/main/human_domainome/SupplementaryTable2.txt?download=true",
            "datasets/xulab-research/TidyMut/resolve/main/human_domainome/SupplementaryTable4.txt?download=true",
            "datasets/xulab-research/TidyMut/resolve/main/human_domainome/wild_type.fasta?download=true",
        ],
        "file_name": [
            "SupplementaryTable2.txt",
            "SupplementaryTable4.txt",
            "wild_type.fasta",
        ],
    },
}


def list_datasets_with_built_in_cleaners() -> None:
    """
    List built-in datasets with predefined processing pipelines.

    These are public datasets for which this package includes pre-defined
    data cleaning pipelines. The datasets themselves are not distributed
    with the package and must be downloaded manually.

    You can also define custom cleaner functions for your own datasets using
    the same `@pipeline_step` framework.

    Predefined datasets:

    - cDNAProteolysis
    - ProteinGym
    - HumanDomainome
    """
    print("Public datasets with ready-to-use cleaning pipelines:")
    for key, info in DATASETS.items():
        print(f"- {key}: {info['name']}")
        print(f"  - Official URL: {info['official_url']}")


def show_download_instructions(dataset_key: str) -> None:
    """
    Show download instructions for a specific dataset.
    """
    info = DATASETS.get(dataset_key)
    if not info:
        raise KeyError(f"Dataset key not found: {dataset_key}")

    print(f"Dataset: {info['name']}")
    for i, file in enumerate(info["files"]):
        print(f"  - File: {file}")
        print(f"    - Download link: {info['huggingface_repos'][i]}")
