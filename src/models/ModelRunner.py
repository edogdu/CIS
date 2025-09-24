from abc import ABC, abstractmethod
class ModelRunner(ABC):
    """Abstract base class for running machine learning models with XAI support."""

    @abstractmethod
    def train(self, data):
        """Train the model with the provided data."""
        pass

    @abstractmethod
    def predict(self, data):
        """Make predictions using the trained model."""
        pass

    @abstractmethod
    def explain(self, data):
        """Explain the model's predictions using XAI techniques."""
        pass
    
    @abstractmethod
    def save_model(self, path: str):
        """Save the trained model to the specified path."""
        pass

    @abstractmethod
    def load_model(self, path: str):
        """Load a trained model from the specified path."""
        pass

    @abstractmethod
    def evaluate(self, data, labels):
        """Evaluate the model's performance on the provided data and labels."""
        pass