"""Moving Targets Backends for Masters."""

from moving_targets.masters.backends.backend import Backend
from moving_targets.masters.backends.gurobi_backend import GurobiBackend

aliases: dict = {'gurobi': GurobiBackend}
