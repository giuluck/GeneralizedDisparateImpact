from src.experiments.adult import Adult
from src.experiments.communities import Communities
from src.experiments.experiment import Experiment


def get(alias: str) -> Experiment:
    match alias:
        case 'adult continuous':
            return Adult(continuous=True)
        case 'adult categorical':
            return Adult(continuous=False)
        case 'communities continuous':
            return Communities(continuous=True)
        case 'communities categorical':
            return Communities(continuous=False)
        case _:
            raise AssertionError(f"Unknown experiment alias '{alias}'")
