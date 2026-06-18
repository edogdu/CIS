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
