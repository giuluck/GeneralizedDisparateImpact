from src.experiments.adult import Adult, AdultRace, AdultAge
from src.experiments.communities import Communities, CommunitiesRace, CommunitiesIncome
from src.experiments.experiment import Experiment


def get(alias: str) -> Experiment:
    if alias == 'adult age':
        return AdultAge()
    elif alias == 'adult race':
        return AdultRace()
    elif alias == 'communities race':
        return CommunitiesRace()
    elif alias == 'communities income':
        return CommunitiesIncome()
    else:
        raise AssertionError(f"Unknown experiment alias '{alias}'")
