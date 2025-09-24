from pydantic import Enum, auto

class ModelTypes(str, Enum):
    GNN = auto()
    XBOOST = auto()