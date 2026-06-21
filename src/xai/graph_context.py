# Step 1 - Binary Explainer
class BinaryExplainer:

    def __init__(self, model):
        self.model = model
        self.ig = IntegratedGradients(self.forward)

    def forward(self, data):
        bin_logits, _ = self.model(data)
        return bin_logits

    def explain(self, data):
        attr = self.ig.attribute(data.x_dict, target=1)
        return attr

    # wrapper function
    def _fast_model_forward_wrapper(model: nn.Module, data: HeteroData, model_device: str, *node_inputs: torch.Tensor):
        """
        Optimized wrapper for Captum. Reconstructs the batched graph from interpolated inputs.
        """
        # Create a lightweight container
        temp_data = HeteroData()
        
        # Identify node types with features
        node_types_with_features = [ntype for ntype in data.x_dict.keys() 
                                    if data[ntype].num_nodes > 0 and hasattr(data[ntype], "x")]
        
        # Determine Batching Params
        first_input = node_inputs[0]
        original_num_nodes = data[node_types_with_features[0]].num_nodes
        total_nodes = first_input.size(0)
        
        is_batched = total_nodes != original_num_nodes
        num_replications = total_nodes // original_num_nodes if is_batched else 1
        
        
        temp_data.num_graphs = num_replications
    
        # Reconstruct Node Features & Batch Vector
        for idx, ntype in enumerate(node_types_with_features):
            temp_data[ntype].x = node_inputs[idx]
            
            if is_batched:
                current_num_nodes = data[ntype].num_nodes
                # Create batch vector [0,0... 1,1...]
                batch_ids = torch.arange(num_replications, device=model_device)
                temp_data[ntype].batch = torch.repeat_interleave(batch_ids, current_num_nodes)
            else:
                temp_data[ntype].batch = torch.zeros(data[ntype].num_nodes, dtype=torch.long, device=model_device)
    
        # Reconstruct Edges (Vectorized)
        if not is_batched:
            temp_data.edge_index_dict = data.edge_index_dict
        else:
            new_edge_index_dict = {}
            for etype, edge_index in data.edge_index_dict.items():
                # Standard edge replication logic
                num_edges = edge_index.size(1)
                src_ntype, _, dst_ntype = etype
                src_count = data[src_ntype].num_nodes
                dst_count = data[dst_ntype].num_nodes
                
                # Create offsets
                offsets_src = (torch.arange(num_replications, device=model_device) * src_count).view(-1, 1)
                offsets_dst = (torch.arange(num_replications, device=model_device) * dst_count).view(-1, 1)
                
                # Expand edges [2, num_edges] -> [num_reps, 2, num_edges]
                edges_expanded = edge_index.unsqueeze(0).expand(num_replications, 2, num_edges).clone()
                
                # Add offsets
                edges_expanded[:, 0, :] += offsets_src
                edges_expanded[:, 1, :] += offsets_dst
                
                # Flatten to [2, total_edges]
                new_edge_index_dict[etype] = edges_expanded.permute(1, 0, 2).reshape(2, -1)
                
            temp_data.edge_index_dict = new_edge_index_dict
    
        # Run Model
        return model(temp_data)

# Step 2 - Attack Explainer
class AttackExplainer:

    def __init__(self, model):
        self.model = model
        self.ig = IntegratedGradients(self.forward)

    def forward(self, data):
        _, anom_logits = self.model(data)
        return anom_logits

    def explain(self, data, class_idx):
        attr = self.ig.attribute(data.x_dict, target=class_idx)
        return attr

# Step 3 - Node Aggregation
# convert feature level to node level
# map it afterward
node_score = attr.mean(dim=feature_dim)

# Step 4 - Graph Context
neighbors = graph.get_neighbors(top_nodes)

# Step 5 is in separate file for generating reports...
