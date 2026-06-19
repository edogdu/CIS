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
