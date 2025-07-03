# tidymut/utils/data_source.py

DATASETS = {
    "K50": {
        "name": "Mega-scale experimental analysis of protein folding stability in biology and design",
        "url": "https://zenodo.org/records/7992926",
        "file": "'Tsuboyama2023_Dataset2_Dataset3_20230416.csv' in 'Processed_K50_dG_datasets.zip'",
    },
    "ProteinGym": {
        "name": "ProteinGym",
        "url": "https://proteingym.org/download",
        "file": "DMS_ProteinGym_substitutions.zip",
    },
    "HumanDomainome": {
        "name": "Site-saturation mutagenesis of 500 human protein domains",
        "url": "https://www.nature.com/articles/s41586-024-08370-4",
        "file": "SupplementaryTable4.txt",
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
    - K50
    - ProteinGym
    - HumanDomainome
    """
    print("Public datasets with ready-to-use cleaning pipelines:")
    for key, info in DATASETS.items():
        print(f"- {key}: {info['name']}")
        print(f"  - File: {info['file']}")
        print(f"  - URL: {info['url']}")


def show_download_instructions(dataset_key: str) -> None:
    """
    Show download instructions for a specific dataset.
    """
    info = DATASETS.get(dataset_key.upper())
    if not info:
        raise KeyError(f"Dataset key not found: {dataset_key}")

    print(
        f"""
Dataset: {info['name']}
File: {info['file']}
Download URL: {info['url']}

Please download the required file manually from the URL above
"""
    )
