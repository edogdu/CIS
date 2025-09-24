from pydantic import Enum,auto

class XaiTypes(str, Enum):
    SHAP = auto()
    LIME = auto()
    GNNEXPLAINER = auto()
    NONE = auto()
