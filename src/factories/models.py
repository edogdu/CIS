from schemas.ModelTypes import ModelTypes
from schemas.XaiTypes import XaiTypes
from models.gnn_ae import GNNAEModelRunner
from models.xgboost import XGBoostModelRunner
from xai.Lime import LimeExplainer
from xai.Shap import ShapExplainer
from xai.GnnExplainer import GnnExplainer

class ModelRepositoryFactory:
    @staticmethod
    def get_model_repository(model_type: ModelTypes, xai_type: XaiTypes):
        if model_type == ModelTypes.GNN:
            return GNNAEModelRunner(xai_type)
        elif model_type == ModelTypes.XBOOST:
            if xai_type == XaiTypes.GNNEXPLAINER:
                raise ValueError("GNNExplainer is not compatible with XGBoost model")
            return XGBoostModelRunner(xai_type)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")