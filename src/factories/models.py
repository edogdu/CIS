from schemas.ModelTypes import ModelTypes
from schemas.XaiTypes import XaiTypes
from models.gnn_ae import GNNAEModelRunner
from models.xgboost import XGBoostModelRunner
from xai.Lime import LimeExplainer
from xai.Shap import ShapExplainer
from xai.GnnExplainerRunner import GnnExplainerRunner
import logging
import os
import torch

model_log = logging.getLogger("models")


class ModelRepositoryFactory:
    
    
    @staticmethod
    def get_model_runner(model_type: ModelTypes,                          
                           input_dim: int,
                           hidden_dim: int,
                           output_dim: int,
                           xai_type: XaiTypes=XaiTypes.NONE,
                           config: dict = None,
                           device: str = None,
                           load_from_path: bool = False,
                           **kwargs):
        if config is None:
            config = {}
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_log.info(f"Creating model runner for type: {model_type}, device: {device}")        
        if model_type == ModelTypes.GNN:
            if load_from_path:
                model_log.info(f"Loading model from: {load_from_path}")
                return GNNAEModelRunner.load_model(config=config)            
            xai_runner = XaiTypes.NONE
            if xai_type == XaiTypes.LIME:
                xai_runner = LimeExplainer()
            elif xai_type == XaiTypes.SHAP:
                xai_runner = ShapExplainer()
            elif xai_type == XaiTypes.GNNEXPLAINER:
                xai_epochs = int(config.get('epochs', 100))
                mask = config.get('xai_feat_mask', 'scalar')
                if mask not in ['scalar', 'feature']:
                    mask = 'scalar'
                model_log.info("xai_epochs: %d, mask: %s, device: %s", xai_epochs, mask, device)

                xai_runner = GnnExplainerRunner(model=None,
                                                epochs=xai_epochs,
                                                lr=0.01,
                                                feat_mask_type=mask,
                                                device=device)
                
                return GNNAEModelRunner(input_dim=input_dim, 
                                    hidden_dim=hidden_dim, 
                                    output_dim=output_dim, 
                                    xai_runner=xai_runner, 
                                    config=config, 
                                    device=device, 
                                    **kwargs)
        elif model_type == ModelTypes.XGBOOST:
            xai_runner = XaiTypes.NONE
            if xai_type == XaiTypes.LIME:
                xai_runner = LimeExplainer()
            elif xai_type == XaiTypes.SHAP:
                xai_runner = ShapExplainer()
            return XGBoostModelRunner(xai_runner=xai_runner, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        