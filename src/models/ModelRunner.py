from abc import ABC, abstractmethod

class ModelRunner(ABC):

    @abstractmethod
    def train(self, train_loader, val_loader):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def evaluate(self, test_loader):
        pass

    @abstractmethod
    def explain(self, data):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def get_criterion(self, train_loader):
        pass
