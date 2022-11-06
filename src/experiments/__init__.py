from src.experiments.adult import Adult, AdultCategorical, AdultContinuous
from src.experiments.communities import Communities, CommunitiesCategorical, CommunitiesContinuous
from src.experiments.experiment import Experiment


def get(alias: str) -> Experiment:
    if alias == 'adult continuous':
        return AdultContinuous()
    elif alias == 'adult categorical':
        return AdultCategorical()
    elif alias == 'communities continuous':
        return CommunitiesContinuous()
    elif alias == 'communities categorical':
        return CommunitiesCategorical()
    else:
        raise AssertionError(f"Unknown experiment alias '{alias}'")
