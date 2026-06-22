# model should only know about: forward(), predict(), train(), evaluate(), save(), load()
# XAI modules handle explanations
import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from repositories.graphs.pyg_builder import y_labels
from models.encoders.hetero_encoder import GNNHeteroEncoderModel

# Local application/library specific imports
logging.info("Imported y_labels in gnn_het.py: %s", y_labels)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

class GNNHeteroClassifierModel(nn.Module):

    def __init__(self, config, metadata):
        super().__init__()

        self.config = config
        self.metadata = metadata
        self.scalers = {}
        self.criterion = None

        self.encoder = GNNHeteroEncoderModel(
            config,
            metadata,
        )

        hd = config.get(
            "hidden_dim",
            64
        )

        self.out = nn.Linear(
            hd,
            len(y_labels)
        )

        self.to(DEVICE)

    # optional helper method for basic explanation
    def explain_snapshot(
        self,
        data,
        save_dir="./exports/explanations",
    ):
        from xai.captum_explainer import CaptumExplainer
        from xai.report_generator import ExplanationReportGenerator
    
        explainer = CaptumExplainer(self)
    
        result = explainer.explain(data)
    
        reporter = ExplanationReportGenerator()
    
        return reporter.generate(
            data,
            result,
            save_dir=save_dir,
        )
