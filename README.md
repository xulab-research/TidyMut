# TidyMut
[![License badge](https://img.shields.io/badge/License-BSD_3--Clause-yellow?logo=opensourceinitiative&logoColor=white)](https://github.com/xulab-research/TidyMut/blob/main/LICENSE)
[![PyPI version badge](https://img.shields.io/pypi/v/tidymut?logo=python&logoColor=white&color=orange)](https://pypi.org/project/tidymut/)
[![Documentation badge](https://img.shields.io/readthedocs/tidymut/latest?logo=readthedocs&logoColor=white)](https://tidymut.readthedocs.io/en/)

A comprehensive Python package for processing and analyzing biological sequence data with advanced mutation analysis capabilities.

- **Documentation**: https://tidymut.readthedocs.io/en/latest
- **Cleaning Examples**: https://tidymut.readthedocs.io/en/latest/user_guide/cleaners.html

## Overview

TidyMut is designed for bioinformaticians, computational biologists, and researchers working with genetic sequence data. The package streamlines the complex process of cleaning, processing, and analyzing DNA and protein sequences, with specialized tools for mutation analysis and large-scale dataset handling.

### Key Capabilities

- **Sequence Data Processing**: Comprehensive support for DNA and protein sequence operations including complementation, transcription, translation, and validation
- **Advanced Mutation Analysis**: Specialized tools for detecting, analyzing, and characterizing genetic mutations with statistical insights
- **Intelligent Data Cleaning**: Automated preprocessing pipelines that handle common data quality issues in biological datasets
- **Flexible Pipeline Architecture**: Modular design allowing custom workflow creation for specific research needs
- **High-Performance Processing**: Optimized for handling large-scale sequence datasets efficiently

## Installation

### Requirements
- Python 3.13+
- pandas
- tqdm

### Install via pip
```bash
pip install tidymut
```

### Development Installation
```bash
git clone https://github.com/xulab-research/TidyMut.git tidymut
cd tidymut
pip install -e .
```

## Quick Start
See [ReadtheDocs](https://tidymut.readthedocs.io/en/latest/user_guide/cleaners.html) for more examples.

### Supported Datasets
| Dataset Name    | Reference                                                                           | File                                               | Link                                               |
| --------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- |
| cDNAProteolysis | Mega-scale experimental analysis of protein folding stability in biology and design | Tsuboyama2023_Dataset2_Dataset3_20230416.csv       | https://zenodo.org/records/7992926                 |
| ProteinGym      | ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction        | DMS_ProteinGym_substitutions.zip                   | https://proteingym.org/download                    |
| HumanDomainome  | Site-saturation mutagenesis of 500 human protein domains                            | SupplementaryTable2.txt or SupplementaryTable4.txt | https://www.nature.com/articles/s41586-024-08370-4 |

### Processing cDNAProteolysis Dataset

Here's a complete example demonstrating TidyMut's capabilities with the cDNAProteolysis mutation dataset:

```python
from tidymut import download_cdna_proteolysis_source_file
from tidymut import cdna_proteolysis_cleaner


# Create cDNAProteolysis cleaning pipeline using TidyMut's default pipeline
cdna_proteolysis_filepath = download_cdna_proteolysis_source_file("dir_path", "file_name")["filename"]

cdna_proteolysis_cleaning_pipeline = cdna_proteolysis_cleaner.create_cdna_proteolysis_cleaner(
    cdna_proteolysis_filepath
)

# Clean and process the dataset 
cdna_proteolysis_cleaning_pipeline, cdna_proteolysis_dataset = \
    cdna_proteolysis_cleaner.clean_cdna_proteolysis_dataset(cdna_proteolysis_cleaning_pipeline)

# Save the processed dataset
cdna_proteolysis_dataset.save("output/cleaned_cdna_proteolysis_data")
```

### Basic Sequence Operations

```python
from tidymut.sequence import DNASequence, ProteinSequence

# DNA sequence analysis
dna = DNASequence("ATGCGATCGTAGC")
print(f"Complement: {dna.complement()}")
print(f"Reverse complement: {dna.reverse_complement()}")
print(f"Translation: {dna.translate()}")
```

## Core Features

### Sequence Data Manipulation
- **Sequence Validation**: Automatic detection and correction of common sequence errors
- **Format Conversion**: Seamless conversion between different sequence formats
- **Batch Processing**: Efficient handling of large sequence collections

### Mutation Analysis
- **Mutation Detection**: Automated identification of point mutations, insertions, and deletions
- **Statistical Analysis**: Comprehensive mutation frequency and distribution statistics
- **Visualization Tools**: Built-in plotting functions for mutation landscapes

### Data Cleaning & Preprocessing
- **Standardization**: Consistent sequence formatting and annotation
- **Duplicate Removal**: Intelligent handling of redundant sequences

### Pipeline Architecture
- **Modular Design**: Mix and match processing components
- **Parallel Processing**: Multi-core support for large datasets
- **Progress Tracking**: Real-time processing status and logging

## Examples and Use Cases

### Custom Processing Pipeline
```python
import pandas as pd

from tidymut.cleaners.basic_cleaners import (
    read_dataset,
    extract_and_rename_columns,
    filter_and_clean_data,
    convert_data_types,
    validate_mutations,
    convert_to_mutation_dataset_format,
)
from tidymut.cleaners.cdna_proteolysis_custom_cleaners import (
    validate_wt_sequence,
    average_labels_by_name,
    subtract_labels_by_wt,
)
from tidymut.core.dataset import MutationDataset
from tidymut.core.pipeline import Pipeline, create_pipeline

dataset = pd.read_csv("path/to/Tsuboyama2023_Dataset2_Dataset3_20230416.csv")

pipeline = create_pipeline(dataset, "cnda_proteolysis_cleaner")
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
        average_labels_by_name,
        name_columns=("name", "mut_info"),
        label_columns="ddG",
    )
    .then(
        validate_wt_sequence,
        name_column="name",
        mutation_column="mut_info",
        sequence_column="mut_seq",
        wt_identifier="wt",
        num_workers=16
    )
    .then(
        subtract_labels_by_wt,
        name_column="name",
        label_columns="ddG",
        mutation_column="mut_info",
        in_place=True,
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
cdna_proteolysis_dataset_df, cdna_proteolysis_ref_seq = clean_result.data
cdna_proteolysis_dataset = MutationDataset.from_dataframe(
    cdna_proteolysis_dataset_df, cdna_proteolysis_ref_seq
)

# Get execution summary
execution_info = pipeline.get_execution_summary()

# Access artifacts
artifacts = pipeline.artifacts

# Save pipeline state
pipeline.save_structured_data("cdna_proteolysis_cleaner_pipeline.pkl")
```

## Citation

If you use TidyMut in your research, please cite:

```bibtex
@software{tidymut,
  title={
    TidyMut: A comprehensive Python package for processing and analyzing 
    biological sequence data with advanced mutation analysis capabilities
  },
  author={YukunR},
  year={2025},
  url={https://github.com/xulab-research/tidymut}
}
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/xulab-research/tidymut/issues)
- **Discussions**: [GitHub Discussions](https://github.com/xulab-research/tidymut/discussions)