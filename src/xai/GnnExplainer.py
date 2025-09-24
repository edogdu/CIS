import torch
from torch_geometric.nn import GNNExplainer
from xai.XaiExplainer import XaiExplainer
    
class GnnExplainer(XaiExplainer):
    def __init__(self, model, epochs=200, lr=0.01, coeffs=None):
        self.explainer = GNNExplainer(
            model,
            epochs=epochs,
            lr=lr,
            coeffs=coeffs
        )

    def explain(self, x, edge_index, target):
        
        node_idx = target  # Assuming target is the node index for which we want explanations
        node_feat_mask, edge_mask = self.explainer.explain_node(node_idx, x, edge_index)
        return node_feat_mask, edge_mask