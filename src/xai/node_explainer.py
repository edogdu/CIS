# node_imp = attr_tensor.abs().sum(dim=1)

class NodeExplainer:

    @staticmethod
    def rank_nodes(
        data,
        ntype,
        attr_tensor,
    ):
        # need to add compute_node_importances(), and rank_nodes()
