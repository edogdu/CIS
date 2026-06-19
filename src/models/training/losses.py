from models.training.losses import FocalLoss

def get_criterion(self, data_loader: DataLoader) -> nn.Module:
        """Get loss function with class weights to handle class imbalance."""
        
        # Gather all labels from the dataset
        labels = []
        for batch in data_loader:
            logging.info("Processing batch with %d graphs for criterion calculation.", batch.num_graphs)
            logging.info("Batch labels: %s", batch.y.view(-1).long().cpu().numpy().tolist())
            # Map labels to their corresponding anomaly classes, offset by -1 for CrossEntropyLoss
            labels.extend(batch.y.view(-1).long().cpu().numpy().tolist())
        
        weights = self.get_weights(labels, min_num_classes=len(y_labels))
        logging.info("Anomaly Classifier Class Weights: %s", weights)
        anom_criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=DEVICE))
        self.criterion = anom_criterion
        return anom_criterion

        #['normal', 'anomaly', 'scan', 'dos', 'mitm', 'physical fault']
        
        # logging.info("Anomaly Classifier Class Weights: %s", weights)
        # anom_criterion = FocalLoss(alpha=weights, gamma=2.0, reduction='mean')
        # self.criterion = anom_criterion
        # return anom_criterion


# build class weights

# build class entropy
