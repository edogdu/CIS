from pydantic import BaseModel
from typing import Optional
from schemas.XaiTypes import XaiTypes
from schemas.ModelTypes import ModelTypes

class DetectAnomalyRequest(BaseModel):
    snapshot_id: Optional[str]
    start_time: str
    end_time: str
    duration: int  # in seconds
    system_id: str    
    threshold: float = 0.5  # Default threshold for anomaly detection
    model_type: ModelTypes = ModelTypes.GNN  # Default model type
    xai_type: XaiTypes = XaiTypes.GNNEXPLAINER  # Default XAI method
    is_train: bool = True  # Whether to train the model or just predict
    export_model: bool = True  # Whether to export the model after training
    export_performance: bool = True  # Whether to export performance metrics