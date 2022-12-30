from src.experiments.adult import Adult
from src.experiments.communities import Communities
from src.experiments.experiment import Experiment


def get(alias: str) -> Experiment:
    if alias == 'adult continuous':
        return Adult(continuous=True)
    elif alias == 'adult categorical':
        return Adult(continuous=False)
    elif alias == 'communities continuous':
        return Communities(continuous=True)
    elif alias == 'communities categorical':
        return Communities(continuous=False)
    else:
        raise AssertionError(f"Unknown experiment alias '{alias}'")
