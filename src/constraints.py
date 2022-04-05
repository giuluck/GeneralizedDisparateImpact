from abc import ABC

from moving_targets.masters.backends import Backend


class Constraint:
    """Abstract interface for a Moving Targets' constraint."""

    def add(self, expression, backend: Backend):
        """Impose the constraint into the model via the backend.
        
        :param expression:
            The left hand side of the constraint.

        :param backend:
            The model backend.
        """
        raise NotImplementedError()


class HalfBoundedInterval(Constraint, ABC):
    """Abstract interface for a half-bounded interval constraint."""

    def __init__(self, value: float):
        """
        :param value:
            The bounding value.
        """

        self.value = value
        """The bounding value."""


class BoundedInterval(Constraint, ABC):
    """Abstract interface for a bounded interval constraint."""

    def __init__(self, lower: float, upper: float):
        """
        :param lower:
            The lower bound.

        :param upper:
            The upper bound.
        """

        self.lower = lower
        """The lower bound."""

        self.upper = upper
        """The upper bound."""


class Free(Constraint):
    """Unconstrained expression."""

    def add(self, expression, backend: Backend):
        return


class Equal(HalfBoundedInterval):
    """Equality constraint (expression == value)."""

    def add(self, expression, backend: Backend):
        backend.add_constraint(expression == self.value)


class Null(Equal):
    """Null constraint (expression == 0)."""

    def __init__(self):
        super(Null, self).__init__(value=0.0)


class Greater(HalfBoundedInterval):
    """Greater or equal to constraint (expression >= value)."""

    def add(self, expression, backend: Backend):
        return backend.add_constraint(expression >= self.value)


class Positive(Greater):
    """Positive constraint (expression >= 0)."""

    def __init__(self):
        super(Positive, self).__init__(value=0.0)


class Lower(HalfBoundedInterval):
    """Lower or equal to constraint (expression <= value)."""

    def add(self, expression, backend: Backend):
        backend.add_constraint(expression <= self.value)


class Negative(Lower):
    """Negative constraint (expression <= 0)."""

    def __init__(self):
        super(Negative, self).__init__(value=0.0)


class Internal(BoundedInterval):
    """Inner interval constraint (lower <= expression <= upper)."""

    def add(self, expression, backend: Backend):
        backend.add_constraints([expression >= self.lower, expression <= self.upper])


class External(BoundedInterval):
    """Outer interval constraint (expression <= lower or expression >= upper)."""

    def add(self, expression, backend: Backend):
        z = 2 * backend.add_binary_variable() - 1
        backend.add_constraint(z * expression >= 0)
        backend.add_indicator_constraint(z, expression=expression <= self.lower, value=-1)
        backend.add_indicator_constraint(z, expression=expression >= self.upper, value=1)


class Smaller(Internal):
    """Smaller absolute value constraint (-threshold <= expression <= threshold, s.t. threshold > 0)."""

    def __init__(self, threshold: float):
        """
        :param threshold:
            The (positive) threshold value.
        """
        assert threshold > 0, f"Positive threshold value expected, got {threshold}"
        super(Smaller, self).__init__(lower=-threshold, upper=threshold)


class Bigger(External):
    """Bigger absolute value constraint (expression <= -threshold or expression >= threshold, s.t. threshold > 0)."""

    def __init__(self, threshold: float):
        """
        :param threshold:
            The (positive) threshold value.
        """
        assert threshold > 0, f"Positive threshold value expected, got {threshold}"
        super(Bigger, self).__init__(lower=-threshold, upper=threshold)
