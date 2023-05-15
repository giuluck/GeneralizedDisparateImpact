"""This script needs to be changed in order to define a custom Weights & Biases configuration for logging."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WandBConfig:
    """Configuration class for custom Weights & Biases logging."""

    entity: str = 'shape-constraints'
    """The Weights&Biases entity name."""

    gedi_calibration: str = 'gedi-calibration'
    """The Weights&Biases project name for the neural calibration experiments."""

    gedi_mt: str = 'gedi-mt'
    """The Weights&Biases project name for the moving targets experiments."""

    gedi_experiments: str = 'gedi-experiments'
    """The Weights&Biases project name for the final experiments."""
