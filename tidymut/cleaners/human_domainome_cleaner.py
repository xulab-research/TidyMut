# tidymut/cleaners/human_domainome_cleaner.py
from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .base_config import BaseCleanerConfig
from .basic_cleaners import (
    read_dataset,
    merge_columns,
    extract_and_rename_columns,
    filter_and_clean_data,
    convert_data_types,
    apply_mutations_to_sequences,
    convert_to_mutation_dataset_format,
)
from .human_domainome_custom_cleaners import (
    process_domain_positions,
    add_sequences_to_dataset,
    extract_domain_sequences,
)
from ..core.dataset import MutationDataset
from ..core.pipeline import Pipeline, create_pipeline
from ..utils.sequence_io import load_sequences, parse_uniprot_header

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

__all__ = [
    "HumanDomainomeCleanerConfig",
    "create_human_domainome_cleaner",
    "clean_human_domainome_dataset",
]


def __dir__() -> List[str]:
    return __all__


# Create module logger
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class HumanDomainomeCleanerConfig(BaseCleanerConfig):
    """Configuration class for HumanDomainome dataset cleaner

    Inherits from BaseCleanerConfig and adds HumanDomainome-specific configuration options.

    Attributes
    ----------
    sequence_dict_path : Union[str, Path]
        Path to the file containing UniProt ID to sequence mapping
    header_parser : Callable[[str], Tuple[str, Dict[str, str]]]
        Parse Header in fasta files and extract relevant information
    column_mapping : Dict[str, str]
        Mapping from source to target column names
    type_conversions : Dict[str, str]
        Data type conversion specifications
    drop_na_columns: List[str]
        List of column names where null values should be dropped
    is_zero_based : bool
        Whether mutation positions are zero-based
    process_workers : int
        Number of workers for parallel processing
    label_columns : List[str]
        List of score columns to process
    primary_label_column : str
        Primary score column for the dataset
    """

    # Path to sequence dictionary file
    sequence_dict_path: Union[str, Path]

    # Header parser function
    header_parser: Callable[[str], Tuple[str, Dict[str, str]]] = parse_uniprot_header

    # Column mapping configuration
    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "uniprot_ID": "name",
            "wt_aa": "wt_aa",
            "mut_aa": "mut_aa",
            "pos": "pos",
            "PFAM_entry": "PFAM_entry",
            "mean_kcalmol_scaled": "label_humanDomainome",
        }
    )

    # Type conversion configuration
    type_conversions: Dict[str, str] = field(
        default_factory=lambda: {"label_humanDomainome": "float"}
    )

    # columns to perfrom dropping NA
    drop_na_columns: List = field(
        default_factory=lambda: ["name", "PFAM_entry", "pos", "wt_aa", "mut_aa"]
    )

    # Processing parameters
    is_zero_based: bool = False  # Human Domainome uses 1-based positions
    process_workers: int = 16

    # Score columns configuration
    label_columns: List[str] = field(default_factory=lambda: ["label_humanDomainome"])
    primary_label_column: str = "label_humanDomainome"

    # Override default pipeline name
    pipeline_name: str = "human_domainome_cleaner"

    def __post_init__(self):
        self.type_conversions.update({"pos": "int", "mut_rel_pos": "int"})
        return super().__post_init__()

    def validate(self) -> None:
        """Validate HumanDomainome-specific configuration parameters

        Raises
        ------
        ValueError
            If configuration is invalid
        """
        # Call parent validation
        super().validate()

        # Validate sequence dictionary path
        if self.sequence_dict_path is not None:
            seq_path = Path(self.sequence_dict_path)
            if not seq_path.exists():
                raise ValueError(
                    f"Sequence dictionary file not found: {self.sequence_dict_path}"
                )

        # Validate score columns
        if not self.label_columns:
            raise ValueError("label_columns cannot be empty")

        if self.primary_label_column not in self.label_columns:
            raise ValueError(
                f"primary_label_column '{self.primary_label_column}' "
                f"must be in label_columns {self.label_columns}"
            )

        # Validate column mapping
        required_mappings = {
            "uniprot_ID",
            "wt_aa",
            "mut_aa",
            "pos",
            "PFAM_entry",
            "mean_kcalmol_scaled",
        }
        missing = required_mappings - set(self.column_mapping.keys())
        if missing:
            raise ValueError(f"Missing required column mappings: {missing}")


def create_human_domainome_cleaner(
    dataset_or_path: Union[str, Path],
    sequence_dict_path: Union[str, Path],
    config: Optional[
        Union[HumanDomainomeCleanerConfig, Dict[str, Any], str, Path]
    ] = None,
) -> Pipeline:
    """Create HumanDomainome dataset cleaning pipeline

    Parameters
    ----------
    dataset_or_path : Union[pd.DataFrame, str, Path]
        Raw HumanDomainome dataset DataFrame or file path to K50 HumanDomainome
        - File: `SupplementaryTable4.txt` from the article
          'Site-saturation mutagenesis of 500 human protein domains'
    sequence_dict_path : Union[str, Path]
        Path to file containing UniProt ID to sequence mapping
    config : Optional[Union[HumanDomainomeCleanerConfig, Dict[str, Any], str, Path]]
        Configuration for the cleaning pipeline. Can be:
        - HumanDomainomeCleanerConfig object
        - Dictionary with configuration parameters (merged with defaults)
        - Path to JSON configuration file (str or Path)
        - None (uses default configuration)

    Returns
    -------
    Pipeline
        The cleaning pipeline

    Raises
    ------
    FileNotFoundError
        If data file or sequence dictionary file not found
    TypeError
        If config has invalid type
    ValueError
        If configuration validation fails

    Examples
    --------
    Basic usage:
    >>> pipeline = create_human_domainome_cleaner(
    ...     "human_domainome.csv",
    ...     "uniprot_sequences.fasta"
    ... )
    >>> pipeline, dataset = clean_human_domainome_dataset(pipeline)

    Custom configuration:
    >>> config = {
    ...     "process_workers": 8,
    ...     "type_conversions": {"label_humanDomainome": "float32"}
    ... }
    >>> pipeline = create_human_domainome_cleaner(
    ...     "human_domainome.csv",
    ...     "sequences.csv",
    ...     config=config
    ... )

    Load configuration from file:
    >>> pipeline = create_human_domainome_cleaner(
    ...     "data.csv",
    ...     "sequences.fasta",
    ...     config="config.json"
    ... )
    """
    seq_path_obj = Path(sequence_dict_path)
    if not seq_path_obj.exists():
        raise FileNotFoundError(
            f"Sequence dictionary file does not exist: {sequence_dict_path}"
        )

    # Handle configuration parameter
    if config is None:
        final_config = HumanDomainomeCleanerConfig(
            sequence_dict_path=sequence_dict_path
        )
    elif isinstance(config, HumanDomainomeCleanerConfig):
        final_config = config
        # Override sequence_dict_path if not set
        if final_config.sequence_dict_path is None:
            final_config.sequence_dict_path = sequence_dict_path
    elif isinstance(config, dict):
        # Partial configuration - merge with defaults
        default_config = HumanDomainomeCleanerConfig(
            sequence_dict_path=sequence_dict_path
        )
        final_config = default_config.merge(config)
    elif isinstance(config, (str, Path)):
        # Load from file
        final_config = HumanDomainomeCleanerConfig.from_json(config)
        # Override sequence_dict_path if not set
        if final_config.sequence_dict_path is None:
            final_config.sequence_dict_path = sequence_dict_path
    else:
        raise TypeError(
            f"config must be HumanDomainomeCleanerConfig, dict, str, Path or None, "
            f"got {type(config)}"
        )

    # Log configuration summary
    logger.info(
        f"HumanDomainome dataset will be cleaned with pipeline: {final_config.pipeline_name}"
    )
    logger.debug(f"Configuration:\n{final_config.get_summary()}")

    # Load sequence dictionary
    seq_dict = _load_sequence_dict(
        final_config.sequence_dict_path, header_parser=final_config.header_parser
    )

    try:
        # Create pipeline
        pipeline = create_pipeline(dataset_or_path, final_config.pipeline_name)

        # Add cleaning steps
        pipeline = (
            pipeline.delayed_then(
                extract_and_rename_columns,
                column_mapping=final_config.column_mapping,
            )
            .delayed_then(
                filter_and_clean_data,
                drop_na_columns=final_config.drop_na_columns,
            )
            .delayed_then(
                convert_data_types,
                type_conversions=final_config.type_conversions,
            )
            .delayed_then(
                process_domain_positions,
            )
            .delayed_then(
                merge_columns,
                columns_to_merge=[
                    final_config.column_mapping.get("uniprot_ID", "uniprot_ID"),
                    "pos",
                ],
                new_column_name="protein_mut_id",
            )
            .delayed_then(
                add_sequences_to_dataset,
                sequence_dict=seq_dict,
                name_column=final_config.column_mapping.get("uniprot_ID", "uniprot_ID"),
            )
            .delayed_then(
                extract_domain_sequences,
                sequence_column="sequence",
                start_pos_column="start_pos",
                end_pos_column="end_pos",
                num_workers=final_config.process_workers,
            )
            .delayed_then(
                apply_mutations_to_sequences,
                sequence_column="sequence",
                name_column=final_config.column_mapping.get("uniprot_ID", "uniprot_ID"),
                mutation_column="mut_info",
                mutation_sep=",",
                is_zero_based=True,  # After process_domain_positions, positions are 0-based
                sequence_type="protein",
                num_workers=final_config.process_workers,
            )
            .delayed_then(
                convert_to_mutation_dataset_format,
                name_column="protein_mut_id",
                mutation_column="mut_info",
                sequence_column="sequence",
                label_column=final_config.primary_label_column,
                is_zero_based=True,  # After process_domain_positions, positions are 0-based
            )
        )

        # Create pipeline based on dataset_or_path type
        if isinstance(dataset_or_path, (str, Path)):
            pipeline.add_delayed_step(read_dataset, 0, file_format="tsv")
        elif not isinstance(dataset_or_path, pd.DataFrame):
            raise TypeError(
                f"dataset_or_path must be pd.DataFrame or str/Path, "
                f"got {type(dataset_or_path)}"
            )

        return pipeline

    except Exception as e:
        logger.error(f"Error in creating HumanDomainome cleaning pipeline: {str(e)}")
        raise RuntimeError(
            f"Error in creating HumanDomainome cleaning pipeline: {str(e)}"
        )


def clean_human_domainome_dataset(
    pipeline: Pipeline,
) -> Tuple[Pipeline, MutationDataset]:
    """Clean HumanDomainome dataset using configurable pipeline

    Parameters
    ----------
    pipeline : Pipeline
        HumanDomainome dataset cleaning pipeline

    Returns
    -------
    Tuple[Pipeline, MutationDataset]
        - Pipeline: The cleaned pipeline
        - MutationDataset: The cleaned HumanDomainome dataset

    Raises
    ------
    RuntimeError
        If pipeline execution fails
    """
    try:
        # Run pipeline
        pipeline.execute()

        # Extract results
        dataset_df, ref_sequences = pipeline.data
        human_domainome_dataset = MutationDataset.from_dataframe(
            dataset_df, ref_sequences
        )

        logger.info(
            f"Successfully cleaned HumanDomainome dataset: "
            f"{len(dataset_df)} mutations from {len(ref_sequences)} proteins"
        )

        return pipeline, human_domainome_dataset

    except Exception as e:
        logger.error(f"Error in running HumanDomainome cleaning pipeline: {str(e)}")
        raise RuntimeError(
            f"Error in running HumanDomainome cleaning pipeline: {str(e)}"
        )


def _load_sequence_dict(
    seq_dict_path: Union[str, Path],
    header_parser: Optional[Callable[[str], Tuple[str, Dict[str, str]]]] = None,
) -> Dict[str, str]:
    """Load UniProt ID to sequence mapping from file

    Parameters
    ----------
    seq_dict_path : Union[str, Path]
        Path to sequence dictionary file (CSV, TSV, or FASTA format)
    header_parser : Optional[Callable], default=None
        Function to parse FASTA headers. If None, uses UniProt parser.
        Only used for FASTA files.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping UniProt IDs to sequences
    """
    # Use the new load_sequences function
    seq_dict = load_sequences(seq_dict_path, header_parser=header_parser)

    logger.info(f"Loaded {len(seq_dict)} sequences from {seq_dict_path}")
    return seq_dict
