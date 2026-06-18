from models.encoders.hetero_encoder import (
    GNNHeteroEncoderModel
)

class GNNHeteroClassifierModel(nn.Module):
    """GNN model for anomaly detection.  It is supervised model,
    which classifies each graph as normal, MITM, DoS, scan, physical fault, anomaly
    This will allow for heterogeneous graphs with different node types.
    """

    def __init__(self, config: Dict[str, Any], metadata=None):
        super(GNNHeteroClassifierModel, self).__init__()
        if metadata is None:
            raise ValueError("Metadata must be provided for heterogeneous graphs.")
        self.metadata = metadata
        self.config = config        
        self.bin_thres = config.get("binary_threshold", 0.35)
        hd = config.get("hidden_dim", 64)
        self.criterion = None
        self.scalers = {

        }
        self.encoder = GNNHeteroEncoderModel(config, metadata)

        self.out = nn.Linear(hd, len(y_labels))   # -> [B,5] (classes: 1..5 shifted to 0..4 in loss)

        self.to(DEVICE)

    def forward(self, x_or_data, edge_index_dict=None) -> torch.Tensor:

        if isinstance(x_or_data, HeteroData):
            data = x_or_data
        else:
            # Assume x_or_data is a dict of node feature tensors
            data = HeteroData()
            for ntype in x_or_data.keys():
                data[ntype].x = x_or_data[ntype]
            data.edge_index_dict = edge_index_dict            

        h = self.encoder(data)
        logits = self.out(h)                   # [B, 5]
        return logits
        
    def extract_timestamp(self, snapshot_id):
        """
        Extracts timestamp from snapshot_id string.
        Format example: 'testbed_system_1_30s_2021-04-19 16:04:30+00:00'
        """
        # Regex to capture YYYY-MM-DD HH:MM:SS
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', str(snapshot_id))
        if match:
            return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        # Fallback for integer timestamps or failures
        return datetime.min

# moving data engineering and pipeline logic back for now
def build_data_loaders(self, dataset: HeteroData):
        """Stratified split of dataset into train and test sets based on graph labels."""
        logging.info("Building data loaders with stratified split for dataset with %d samples...", len(dataset))
        
        logging.info("Creating  binary dataset...")
        # Get labels for stratification
        labels = [data.y.item() for data in dataset]
        class_counts = np.bincount(labels, minlength=len(y_labels))
        logging.info("Original class distribution: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(class_counts)]))
        
        # split dataset for 60% train, 40% validation/final test
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)

        train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))

        # further split test into 20% validation and 20% final test
        splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        test_labels = [labels[i] for i in test_idx]

        relative_val_idx, relative_final_test_idx = next(splitter2.split(np.zeros(len(test_idx)), test_labels))

        # Map relative indices back to original test indices
        val_idx = [test_idx[i] for i in relative_val_idx]
        final_test_idx = [test_idx[i] for i in relative_final_test_idx]
        
        dataset = self.scale_features(dataset, train_idx, val_idx, final_test_idx, node_type='Connections', feature_column_key='Connections')
        dataset = self.scale_features(dataset, train_idx, val_idx, final_test_idx, node_type='Endpoints', feature_column_key='Endpoints')
        dataset = self.scale_features(dataset, train_idx, val_idx, final_test_idx, node_type='Tanks', feature_column_key='Tanks')
        dataset = self.scale_features(dataset, train_idx, val_idx, final_test_idx, node_type='FlowSensors', feature_column_key='FlowSensors')
        #dataset = self.scale_features(dataset, train_idx, val_idx, final_test_idx, node_type='Pumps', feature_column_key='Pumps')
        #dataset = self.scale_features(dataset, train_idx, val_idx, final_test_idx, node_type='Valves', feature_column_key='Valves')
        #dataset = self.scale_features(dataset, train_idx, val_idx, final_test_idx, node_type='Assets', feature_column_key='Assets')
        #visualize_features_distribution(dataset)
        
        
         
        train_set = [dataset[i] for i in train_idx]
        val_set = Subset(dataset, val_idx)
        final_test_set = Subset(dataset, final_test_idx)

        
        #create loaders
        # weights = self.get_weights([labels[i] for i in train_idx], min_num_classes=len(y_labels))
        # sample_weights = torch.tensor([weights[labels[i]] for i in train_idx], dtype=torch.double)
        # sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)        

        
        # train_loader = DataLoader(train_set, batch_size=32, shuffle=False, num_workers=0, sampler=sampler, pin_memory=True)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
        final_test_loader = DataLoader(final_test_set, batch_size=32, shuffle=False, num_workers=0)

        #Create export csv of split counts for each class for each subset
        logging.info("Exporting data split counts to ./exports/results/gnn_het_data_split_counts.csv")
        original_counts = {
            'Overall': np.bincount(labels, minlength=len(y_labels)),
            'Train': np.bincount([labels[i] for i in train_idx], minlength=len(y_labels)),
            'Validation': np.bincount([labels[i] for i in val_idx], minlength=len(y_labels)),
            'Final Test': np.bincount([labels[i] for i in final_test_idx], minlength=len(y_labels)),
        }
        with open("./exports/results/gnn_het_data_split_counts.csv", "w") as f:
            f.write("Class," + ",".join(y_labels) + "\n")
            for split, counts in original_counts.items():
                f.write(split + "," + ",".join(str(count) for count in counts) + "\n")
        return (train_loader, val_loader), final_test_loader 
 
    def scale_features(self, dataset, train_idx, val_idx, final_test_idx, node_type,feature_column_key):
        ignore_scaling_keywords = ['_bucket_', 'asset_type_', 'protocol_', 'mac_byte_', 'ip_part_',
                                   'response_present']
        log_candidates = [
            'avg_size', 'avg_value','stddev_value', 'min_value','max_value', 'num_connections', 'modbus_response_count',
            'endpoint_unique_peer_count', 'endpoint_num_unique_protocols',
            'tcp_cwr_count', 'tcp_ece_count', 'tcp_urg_count', 'tcp_ack_count', 
            'tcp_psh_count', 'tcp_rst_count', 'tcp_syn_count', 'tcp_fin_count',
            'endpoint_in_out_ratio','endpoint_num_unique_ports', 'endpoint_port_entropy',
        ]

        if node_type not in self.scalers:
            self.scalers[node_type] = {}

        connection_features = get_hetero_column_names(feature_column_key)
        for feature_name in [f for f in connection_features if not any(keyword in f for keyword in ignore_scaling_keywords)]:
            feature_idx = connection_features.index(feature_name)
                        
            # gather all values for this feature from training set
            values = []
            for idx in train_idx:
                data = dataset[idx]
                if node_type in data.x_dict:
                    values.append(data.x_dict[node_type][:, feature_idx].cpu().numpy())
            values = np.concatenate(values)
            # apply log1p scaling to candidates
            if feature_name in log_candidates:
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                values = np.log1p(np.maximum(values, 0))
            scaler = RobustScaler()
            scaler.fit(values.reshape(-1, 1))
            self.scalers[node_type][feature_name] = scaler

            for idx in train_idx:
                data = dataset[idx]
                if node_type in data.x_dict:
                    
                    data.x_dict[node_type][:, feature_idx] = torch.where(data.x_dict[node_type][:, feature_idx] < 0.01, torch.tensor(0.0, device=data.x_dict[node_type].device), data.x_dict[node_type][:, feature_idx])
                    data.x_dict[node_type][:, feature_idx] = torch.log1p(torch.maximum(data.x_dict[node_type][:, feature_idx], torch.tensor(0.0, device=data.x_dict[node_type].device)))
                    data.x_dict[node_type][:, feature_idx] = torch.from_numpy(scaler.transform(data.x_dict[node_type][:, feature_idx].cpu().numpy().reshape(-1, 1))).to(data.x_dict[node_type].device).float().to(data.x_dict[node_type].device).view(-1)
                    
            # apply scaling to val and final test sets
            for idx in val_idx + final_test_idx:
                data = dataset[idx]
                if node_type in data.x_dict:                    
                    data.x_dict[node_type][:, feature_idx] = torch.where(data.x_dict[node_type][:, feature_idx] < 0.01, torch.tensor(0.0, device=data.x_dict[node_type].device), data.x_dict[node_type][:, feature_idx])
                    data.x_dict[node_type][:, feature_idx] = torch.log1p(torch.maximum(data.x_dict[node_type][:, feature_idx], torch.tensor(0.0, device=data.x_dict[node_type].device)))
                    data.x_dict[node_type][:, feature_idx] = torch.from_numpy(scaler.transform(data.x_dict[node_type][:, feature_idx].cpu().numpy().reshape(-1, 1))).to(data.x_dict[node_type].device).float().to(data.x_dict[node_type].device).view(-1)
                    
        return dataset
           
