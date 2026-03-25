# Data Cleaners Usage Guide

## Overview

This guide provides usage examples for data cleaning modules organized by database:

- [**HumanDomainome**](#human-domainome-database): Site-saturation mutagenesis of 500 human protein domains
    - [**Supplementary Table 2**](#supplementarytable2-cleaner): Fitness scores and errors.
    - [**Supplementary Table 4**](#supplementarytable4-cleaner): Homolog-averaged ∆∆G predictions across families mapped to homologous domains proteome-wide.
- [**ProteinGym**](#protein-gym-database): ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction
- [**cDNAProteolysis**](#cdna-proteolysis-database): Mega-scale experimental analysis of protein folding stability in biology and design
- [**ddG-dTm Datasets**](#ddg-dtm-datasets): A collection of datasets providing single- and multiple-mutant measurements, labeled by thermodynamic parameters (ΔΔG, ΔTm)
- [**ArchStabMS1E10 Datasets**](#archstabms1e10-datasets): High-order multi-mutant libraries (“1e10”) measuring protein stability for GRB2-SH3 and SRC.
- [**Antitoxin ParD3 Datasets**](#antitoxin-pard3): The antitoxin ParD3 3-position library is a combinatorially exhaustive dataset of 8,000 variants demonstrating that simple, independent per-residue mutation preferences are sufficient to almost perfectly predict combinatorial protein fitness.
- [**TrpB Datasets**](#trpb-datasets): a combinatorially complete sequence-fitness landscape comprising 160,000 variants across four active-site residues of the enzyme tryptophan synthase, capturing significant epistatic interactions to serve as a benchmark for model-guided enzyme engineering.
- [**Human Myoglobin Datasets**](#human-myoglobin-datasets): a deep mutational scanning library detailing the expression fitness scores for near-comprehensive single-codon and small-fraction double-codon mutations in yeast surface-displayed human myoglobin, which was used to train machine learning models for predicting epistatic effects and discovering stability-enhancing variants.
- [**CTXM Datasets**](#ctxm-database): a comprehensive deep mutational scanning library of 49,096 pairwise double mutations across 17 active site residues of the CTX-M-14 $\beta$-lactamase enzyme, constructed to systematically map the epistatic interaction network driving antibiotic resistance.
    - [**CTXM ampicillin**]: A subset of the CTX-M library quantifying the functional fitness and epistatic interactions of the enzyme variants under ampicillin selection, revealing a broader mutational tolerance and distinct compensatory pathways.
    - [**CTXM cefotaxime**]: A subset of the CTX-M library quantifying the functional fitness and epistatic interactions of the enzyme variants under cefotaxime selection, characterized by highly stringent sequence requirements and substrate-specific epistasis.
- [**RBD ACE2 Database**](#rbd-ace2-database): SARS-CoV-2 RBD sequences with ACE2 binding affinity scores, labeled by `log10Ka` where higher values indicate stronger ACE2 binding affinity.
- [**RBD Antibody Database**](#rbd-antibody-database): SARS-CoV-2 RBD antibody binding data with `score` transformed by negative logarithm. Higher scores indicate stronger binding.


## Prerequisites

```bash
pip install tidymut
```

---

## Human Domainome Database

### File Preparation
You can download the source file directy by running (see {py:func}`tidymut.utils.download_human_domainome_source_file` for details):
```python
from tidymut import download_human_domainome_source_file
filepaths = download_human_domainome_source_file("path/to/target/folder")
```

Alternatively, you can download it from [Nature](https://www.nature.com/articles/s41586-024-08370-4) or [Hugging Face](https://huggingface.co/datasets/xulab-research/TidyMut/tree/main/human_domainome) (See `SupplementaryTable2.txt` or `SupplementaryTable4.txt`)

The Hugging Face dataset already includes the reference FASTA. If you are not using that source, you’ll need to provide the FASTA yourself (i.e., the reviewed Human (9606) proteome from  [UNIPROT](
https://rest.uniprot.org/uniprotkb/stream?download=true&format=fasta&query=%28*%29+AND+%28model_organism%3A9606%29+AND+%28reviewed%3Atrue%29)).

### SupplementaryTable2 Cleaner

**Cleaning Pipeline**

```python
from tidymut.cleaners import (
    create_human_domainome_sup2_cleaner, 
    clean_human_domainome_sup2_dataset
)

# File settings
dataset_filepath = "path/to/dataset/file"
# Clean data
hd_cleaning_pipeline = create_human_domainome_sup2_cleaner(dataset_filepath)
hd_cleaning_pipeline, hd_dataset = clean_human_domainome_sup2_dataset(hd_cleaning_pipeline)
```

**Advanced Settings**

See {py:func}`tidymut.cleaners.HumanDomainomeSup2CleanerConfig` for details.

### SupplementaryTable4 Cleaner

**Cleaning Pipeline**

```python
from tidymut.cleaners import (
    create_human_domainome_sup4_cleaner, 
    clean_human_domainome_sup4_dataset
)

# File settings
dataset_filepath = "path/to/dataset/file"
reference_fasta_filepath = "path/to/fasta"
# Clean data
hd_cleaning_pipeline = create_human_domainome_sup4_cleaner(dataset_filepath, reference_fasta_filepath)
hd_cleaning_pipeline, hd_dataset = clean_human_domainome_sup4_dataset(hd_cleaning_pipeline)
```

**Advanced Settings**

See {py:func}`tidymut.cleaners.HumanDomainomeSup4CleanerConfig` for details.

---

## Protein Gym Database

### File Preparation
You can download the source file directy by running (see {py:func}`tidymut.utils.download_protein_gym_source_file` for details):
```python
from tidymut import download_protein_gym_source_file
filepaths = download_protein_gym_source_file("path/to/target/folder")
```

Alternatively, you can download it from [ProteinGym](https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip) or [Hugging Face](https://huggingface.co/datasets/xulab-research/TidyMut/tree/main/ProteinGym_DMS_substitutions)

### Basic Usage

**Cleaning Pipeline**

```python
from tidymut.cleaners import (
    create_protein_gym_cleaner,
    clean_protein_gym_dataset
)

# File settings
dataset_filepath = "path/to/dataset/file"
# Clean data
pg_cleaning_pipeline = create_protein_gym_cleaner(dataset_filepath)
pg_cleaning_pipeline, pg_dataset = clean_protein_gym_dataset(pg_cleaning_pipeline)
```

**Advanced Settings**

See {py:func}`tidymut.cleaners.ProteinGymCleanerConfig` for details.

## cDNA Proteolysis Database

### File Preparation
You can download the source file directy by running (see {py:func}`tidymut.utils.download_cdna_proteolysis_source_file` for details):
```python
from tidymut import download_cdna_proteolysis_source_file
filepaths = download_cdna_proteolysis_source_file("path/to/target/folder")
```

Alternatively, you can download it from [Zenodo](https://zenodo.org/records/7992926) ("'Tsuboyama2023_Dataset2_Dataset3_20230416.csv' in 'Processed_K50_dG_datasets.zip'") or [Hugging Face](https://huggingface.co/datasets/xulab-research/TidyMut/tree/main/cDNA_proteolysis)

### ΔΔG as Label (Default Pipeline)

**Cleaning Pipeline**

```python
from tidymut.cleaners import (
    create_cdna_proteolysis_cleaner,
    clean_cdna_proteolysis_dataset
)

# File settings
dataset_filepath = "path/to/dataset/file"
# Clean data
cdnap_cleaning_pipeline = create_cdna_proteolysis_cleaner(dataset_filepath)
cdnap_cleaning_pipeline, cdnap_dataset = clean_cdna_proteolysis_dataset(cdnap_cleaning_pipeline)
```

### ΔG as Label

**Cleaning Pipeline**
```python
from tidymut.cleaners import (
    CDNAProteolysisCleanerConfig, 
    create_cdna_proteolysis_cleaner,
    clean_cdna_proteolysis_dataset
)

# File settings
dataset_filepath = "path/to/dataset/file"

# Set cleaning configs
cdnap_cleaning_config = CDNAProteolysisCleanerConfig()
cdnap_cleaning_config.column_mapping = {
    "WT_name": "name",
    "aa_seq": "mut_seq",
    "mut_type": "mut_info",
    "dG_ML": "label_cDNAProteolysis",
}
# Clean data
cdnap_cleaning_pipeline = create_cdna_proteolysis_cleaner(dataset_filepath, cdnap_cleaning_config)
cdnap_cleaning_pipeline, cdnap_dataset = clean_cdna_proteolysis_dataset(cdnap_cleaning_pipeline)
```

## ddG-dTm Datasets

### File Preparation

You can download the source file directy by running (see {py:func}`tidymut.utils.download_ddg_dtm_source_file` for details):
```python
from tidymut import download_ddg_dtm_source_file

# Download all datasets
filepaths = download_ddg_dtm_source_file("path/to/target/folder")

# Or specify a particular dataset, e.g.
filepath = download_ddg_dtm_source_file("path/to/target/folder", sub_dataset = "S571")
```

### Basic Usage

{py:func}`tidymut.cleaners.ddg_dtm_cleaners.create_ddg_dtm_cleaner` can automatically recognize the label column (ddG or dTm). For example:

```python
from tidymut.cleaners import (
    create_ddg_dtm_cleaner,
    clean_ddg_dtm_dataset
)

# File settings
dataset_filepath = "path/to/dataset/file"

# Clean data
ddgdtm_cleaning_pipeline = create_ddg_dtm_cleaner(dataset_filepath)
ddgdtm_cleaning_pipeline, ddgdtm_dataset = clean_ddg_dtm_dataset(ddgdtm_cleaning_pipeline)
```

### Advanced Settings

See {py:func}`tidymut.cleaners.DdgDtmCleanerConfig` for details.

## ArchStabMS1E10 Datasets

### File Preparation

You can download the source file directy by running (see {py:func}`tidymut.utils.download_archstabms1e10_source_file` for details):
```python
from tidymut import download_archstabms1e10_source_file
filepaths = download_archstabms1e10_source_file("path/to/target/folder")
```

### Basic Usage

```python
from tidymut.cleaners import (
    create_archstabms_1e10_cleaner,
    clean_archstabms_1e10_dataset
)

# File settings
dataset_filepath = "path/to/dataset/file"

# Clean data
archstabms_cleaning_pipeline = create_archstabms_1e10_cleaner(dataset_filepath)
archstabms_cleaning_pipeline, archstabms_dataset = clean_archstabms_1e10_dataset(ddgdtm_cleaning_pipeline)
```

### Advanced Settings

See {py:func}`tidymut.cleaners.ArchStabMS1E10CleanerConfig` for details.

## Antitoxin ParD3

### File Preparation

You can download the source file directy by running (see {py:func}`tidymut.utils.download_antitoxin_pard3_source_file` for details):
```python
from tidymut import download_antitoxin_pard3_source_file
filepaths = download_antitoxin_pard3_source_file("path/to/target/folder")
```

### Basic Usage

```python
from tidymut.cleaners import (
    create_antitoxin_pard3_cleaner,
    clean_antitoxin_pard3_dataset
)

# File settings
dataset_filepath = "path/to/dataset/file"

# Clean data
antitoxin_pard3_cleaning_pipeline = create_antitoxin_pard3_cleaner(dataset_filepath)
antitoxin_pard3_cleaning_pipeline, antitoxin_pard3_dataset = clean_antitoxin_pard3_dataset(ddgdtm_cleaning_pipeline)
```

### Advanced Settings

See {py:func}`tidymut.cleaners.AntitoxinParD3CleanerConfig` for details.

## TrpB Datasets

### File Preparation

You can download the source file directy by running (see {py:func}`tidymut.utils.download_trpb_source_file` for details):
```python
from tidymut import download_trpb_source_file
filepaths = download_trpb_source_file("path/to/target/folder")
```

### Basic Usage

```python
from tidymut.cleaners import (
    create_trpb_cleaner,
    clean_trpb_dataset
)

# File settings
dataset_filepath = "path/to/dataset/file"

# Clean data
trpB_cleaning_pipeline = create_trpb_cleaner(dataset_filepath)
trpB_cleaning_pipeline, trpB_dataset = clean_trpb_dataset(trpB_cleaning_pipeline)
```

### Advanced Settings

See {py:func}`tidymut.cleaners.TrpBCleanerConfig` for details.

## Human Myoglobin Datasets


### File Preparation

You can download the source file directy by running (see {py:func}`tidymut.utils.download_human_myoglobin_source_file` for details):
```python
from tidymut import download_human_myoglobin_source_file
filepaths = download_human_myoglobin_source_file("path/to/target/folder")
```

### Basic Usage

```python
from tidymut.cleaners import (
    create_trpb_cleaner,
    clean_trpb_dataset
)

# File settings
dataset_filepath = "path/to/dataset/file"

# Clean data
human_myoglobin_cleaning_pipeline = create_human_myoglobin_cleaner(dataset_filepath)
human_myoglobin_cleaning_pipeline, human_myoglobin_dataset = clean_trpb_dataset(human_myoglobin_cleaning_pipeline)
```

### Advanced Settings

See {py:func}`tidymut.cleaners.HumanMyoglobinCleanerConfig` for details.

## CTXM DataBase


### File Preparation

You can download the source file directy by running (see {py:func}`tidymut.utils.download_ctxm_source_file` for details):
```python
from tidymut import download_ctxm_source_file
filepaths = download_ctxm_source_file("path/to/target/folder")
```

### Basic Usage

```python
from tidymut.cleaners import (
    create_ctxm_cleaner,
    clean_ctxm_dataset,
)

# File settings
dataset_filepath = "path/to/dataset/file"

# Clean data
ctxm_cleaning_pipeline = create_ctxm_cleaner(dataset_filepath)
ctxm_cleaning_pipeline, ctxm_dataset = clean_trpb_dataset(ctxm_cleaning_pipeline)
```

### Advanced Settings

See {py:func}`tidymut.cleaners.CTXMCleanerConfig` for details.


## RBD ACE2 Database

### File Preparation

You can download the source file directly by running (see {py:func}`tidymut.utils.download_rbd_ace2_source_file` for details):
```python
from tidymut import download_rbd_ace2_source_file
filepaths = download_rbd_ace2_source_file("path/to/target/folder")
```

You can also download and process a specific sub-dataset:

```python
from tidymut import download_rbd_ace2_source_file
filepaths = download_rbd_ace2_source_file(
    "path/to/target/folder",
    sub_dataset="Omicron_EG5_FLip_BA286",
)
```

Supported sub-datasets:
- `Omicron_EG5_FLip_BA286`
- `Omicron_XBB_BQ`
- `Omicron`
- `DMS_variants`
- `Delta`

Alternatively, you can download it from [Hugging Face](https://huggingface.co/datasets/Zoey13891350636/RBD_ACE2).

### Basic Usage

```python
from tidymut import download_rbd_ace2_source_file
from tidymut import rbd_ace2_cleaner

rbd_ace2_filepaths = download_rbd_ace2_source_file(
    "path/to/target/folder",
    sub_dataset="Omicron_EG5_FLip_BA286",
)
dataset_filepath = next(iter(rbd_ace2_filepaths.values()))

rbd_ace2_cleaning_pipeline = rbd_ace2_cleaner.create_rbd_ace2_cleaner(dataset_filepath)
rbd_ace2_cleaning_pipeline, rbd_ace2_dataset = rbd_ace2_cleaner.clean_rbd_ace2_dataset(
    rbd_ace2_cleaning_pipeline
)
```

### Advanced Settings

See {py:func}`tidymut.cleaners.RBDACE2CleanerConfig` for details.

## RBD Antibody Database

### File Preparation

You can download the source file directly by running (see {py:func}`tidymut.utils.download_rbd_antibody_source_file` for details):
```python
from tidymut import download_rbd_antibody_source_file
filepaths = download_rbd_antibody_source_file("path/to/target/folder")
```

You can also download and process a specific sub-dataset:

```python
from tidymut import download_rbd_antibody_source_file
filepaths = download_rbd_antibody_source_file(
    "path/to/target/folder",
    sub_dataset="AZ_Abs",
)
```

Supported sub-datasets:
- `AZ_Abs`
- `HAARVI_sera`
- `Moderna`
- `Rockefeller`
- `Vir_mAbs`
- `clinical_Abs`

Alternatively, you can download it from [Hugging Face](https://huggingface.co/datasets/Zoey13891350636/RBD_Antibody).

### Basic Usage

```python
from tidymut import download_rbd_antibody_source_file
from tidymut import rbd_antibody_cleaner

filepaths = download_rbd_antibody_source_file(
    "path/to/target/folder",
    sub_dataset="AZ_Abs",
)
dataset_filepath = next(iter(filepaths.values()))

rbd_cleaning_pipeline = rbd_antibody_cleaner.create_rbd_antibody_cleaner(dataset_filepath)
rbd_cleaning_pipeline, rbd_dataset = rbd_antibody_cleaner.clean_rbd_antibody_dataset(
    rbd_cleaning_pipeline
)
```

### Advanced Settings

See {py:func}`tidymut.cleaners.RBDAntibodyCleanerConfig` for details.
