"""Abstract base class for CMA-ES-style optimizers."""

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Minimal ask-tell interface for black-box optimizers.

    Subclasses implement the ``ask`` / ``tell`` loop used by
    ``CMAESWorkflow``.  The design follows the standard CMA-ES
    ask-tell API (as in the ``cmaes`` Python package).
    """

    @abstractmethod
    def ask(self):
        """Sample a new population of candidate parameter vectors.

        Returns:
            Sequence of candidate vectors (one per individual).
        """
        pass

    @abstractmethod
    def tell(self, solutions):
        """Update the optimizer with evaluated solutions.

        Args:
            solutions: Sequence of ``(parameters, score)`` pairs from the
                current generation, where lower scores are better.
        """
        pass