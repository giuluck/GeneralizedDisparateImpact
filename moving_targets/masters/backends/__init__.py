"""Moving Targets Backends for Masters."""

from moving_targets.masters.backends.backend import Backend
from moving_targets.masters.backends.gurobi_backend import GurobiBackend
from moving_targets.masters.backends.numpy_backend import NumpyBackend

aliases: dict = {'gurobi': GurobiBackend}
