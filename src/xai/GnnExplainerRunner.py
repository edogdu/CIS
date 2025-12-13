import torch
from torch import nn
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain import Explainer, Explanation
from xai.XaiExplainer import XaiExplainer


class LinkPrediction(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, target_edge, **kwargs):
        # target_edge: tuple (u,v) or a [2] LongTensor
        if isinstance(target_edge, tuple):
            u, v = target_edge
            u = torch.tensor([u], device=x.device, dtype=torch.long)
            v = torch.tensor([v], device=x.device, dtype=torch.long)
        else:
            u = target_edge[0].view(1).to(x.device)
            v = target_edge[1].view(1).to(x.device)

        # Encode node features to get embeddings
        z = self.encoder(x, edge_index)
        
        # return logits
        logit = (z[u] * z[v]).sum(dim=-1)
        return logit


class GnnExplainerRunner(XaiExplainer):
    def __init__(self, 
                 model:nn.Module, 
                 epochs=200, 
                 lr=0.01, 
                 feat_mask_type='scalar', 
                 device: torch.device | str | None = None):
        
        self.encoder_model = model
        self.epochs = epochs
        self.lr = lr
        self.feat_mask_type = feat_mask_type
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_encoder(model)
    
    def set_encoder(self, model:nn.Module):
        
        # if the model has an attribute 'encoder', use it; otherwise, use the model itself
        encoder = getattr(model, 'encoder', model)

        self.encoder_model = encoder

        algo = GNNExplainer(            
            epochs=self.epochs,
            lr=self.lr,
            feat_mask_type=self.feat_mask_type,
        )
        self.explainer = Explainer(model=LinkPrediction(self.encoder_model).to(self.device), 
                                        algorithm=algo,
                                        explanation_type='model',
                                        node_mask_type='attributes',
                                        edge_mask_type='object',
                                        model_config=dict(
                                            mode='binary_classification',
                                            task_level='edge',
                                            return_type='raw'
                                        ))

    def explain(self, x:torch.Tensor, edge_index:torch.Tensor, target_edge):

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        explaination = self.explainer(x=x, edge_index=edge_index, target_edge=target_edge)
        node_feat_mask = explaination.get('node_mask', None)
        edge_mask = explaination.get('edge_mask', None)
        return {
            'node_feat_mask': node_feat_mask.detach().cpu(),
            'edge_mask': edge_mask.detach().cpu()
        }