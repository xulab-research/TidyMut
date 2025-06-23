# TidyMut

A comprehensive Python package for processing and analyzing biological sequence data with advanced mutation analysis capabilities.

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

### Processing K50 Dataset

Here's a complete example demonstrating TidyMut's capabilities with the K50 mutation dataset:

```python
import pandas as pd
from tidymut.cleaners.k50_cleaner import clean_k50_dataset

# Load the K50 dataset
# Download from: https://zenodo.org/records/799292
# File: `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` in `Processed_K50_dG_datasets.zip`
raw_data = pd.read_csv("path/to/Tsuboyama2023_Dataset2_Dataset3_20230416.csv")

# Clean and process the dataset using TidyMut's default pipeline
k50_dataset = clean_k50_dataset(raw_data)

# Save the processed dataset
k50_dataset.save("output/cleaned_k50_data")

# Access processed data
print(f"Dataset contains {len(k50_dataset)} sequences")
print(f"Mutation types identified: {k50_dataset.mutation_summary()}")
```

### Basic Sequence Operations

```python
from tidymut.sequence import DNASequence, ProteinSequence

# DNA sequence analysis
dna = DNASequence("ATGCGATCGTAGC")
print(f"Complement: {dna.complement()}")
print(f"Reverse complement: {dna.reverse_complement()}")
print(f"Translation: {dna.translate()}")

# Protein sequence analysis
protein = ProteinSequence("MRSIVA")
print(f"Molecular weight: {protein.molecular_weight()}")
print(f"Hydrophobicity: {protein.hydrophobicity_score()}")
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

### Comparative Mutation Analysis
```python
from tidymut.analysis import MutationComparator

comparator = MutationComparator()
comparator.add_dataset("wildtype", wt_sequences)
comparator.add_dataset("variant", variant_sequences)

results = comparator.compare_mutation_profiles()
comparator.plot_comparison(results)
```

### Custom Processing Pipeline
```python
import pandas as pd
from typing import Tuple

from tidymut.cleaners.basic_cleaners import (
    extract_and_rename_columns,
    filter_and_clean_data,
    convert_data_types,
    validate_mutations,
    infer_wildtype_sequences,
    convert_to_mutation_dataset_format,
)
from tidymut.core.dataset import MutationDataset
from tidymut.core.pipeline import Pipeline, create_pipeline

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

# Get execution summary
execution_info = pipeline.get_execution_summary()

# Access artifacts
artifacts = pipeline.artifacts

# Save pipeline state
pipeline.save_structured_data("k50_cleaner_pipeline.pkl")
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## Citation

If you use TidyMut in your research, please cite:

```bibtex
@software{tidymut,
  title={TidyMut: A Python Package for Biological Sequence Data Processing},
  author={Your Name and Contributors},
  year={2024},
  url={https://github.com/xulab-research/tidymut}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/xulab-research/tidymut/issues)
- **Discussions**: [GitHub Discussions](https://github.com/xulab-research/tidymut/discussions)
- **Email**: 