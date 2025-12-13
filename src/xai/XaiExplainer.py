from abc import ABC, abstractmethod

class XaiExplainer(ABC):
    """Abstract base class for XAI explainers."""

    @abstractmethod
    def explain(self, instance):
        """Generate an explanation for the given instance."""
        pass