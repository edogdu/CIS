# xai/base.py

from abc import ABC, abstractmethod

class XAIExplainer(ABC):

    @abstractmethod
    def explain(self, model, graph):
        pass
