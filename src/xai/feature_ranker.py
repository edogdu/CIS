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
      
    # Training and evaluation functions

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer):
        self.train()
        total_loss = 0
        
        try:
            total_num = 0
            y_all = []
            y_pred_all = []
            
            for batch in loader:
                batch = batch.to(DEVICE) 
                optimizer.zero_grad()               
                # Inside train_epoch, right before anom_logits = self(batch)
                # if np.random.rand() < 0.01: # Check 1% of batches
                #     logging.info(f"DEBUG CHECK - Batch y sum: {batch.y.sum().item()}")
                #     logging.info(f"DEBUG CHECK - Connection feature mean: {batch['Connections'].x.mean(dim=0)}")
                #     logging.info(f"DEBUG CHECK - Endpoint feature mean: {batch['Endpoints'].x.mean(dim=0)}")
                #     logging.info(f"DEBUG CHECK - Sensor feature mean: {batch['FlowSensors'].x.mean(dim=0)}")
                #     logging.info(f"DEBUG CHECK - Pump feature mean: {batch['Pumps'].x.mean(dim=0)}")
                #     logging.info(f"DEBUG CHECK - Valve feature mean: {batch['Valves'].x.mean(dim=0)}")
                #     logging.info(f"DEBUG CHECK - Tank feature mean: {batch['Tanks'].x.mean(dim=0)}")
                    #logging.info(f"DEBUG CHECK - Asset feature mean: {batch['Assets'].x.mean(dim=0)}")
                anom_logits = self(batch)      # [B,5]
                y = batch.y.view(-1).long()
                #loss = self.criterion(anom_logits, y)
                loss = self.criterion(anom_logits, y)

                # Add L1 regularization to the loss 
                # l1_lambda = 5e-5
                # l1_norm = sum(p.abs().sum() for p in self.parameters())
                # loss = loss + l1_lambda * l1_norm

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                total_num += y.size(0)

                pred = anom_logits.argmax(dim=1)


                y_all.extend(y.detach().cpu().tolist())
                y_pred_all.extend(pred.detach().cpu().tolist())
            # Apply gradient after accumulating over the batch to help with stability
            # This allows for larger effective batch sizes and ensures that
            # smaller classes are represented in each optimization step.
            
            mean_loss = total_loss / max(1, total_num)
            return self.get_label_metrics(y_all, y_pred_all, mean_loss)
        except Exception as e:
            logging.error("Error during training epoch: %s", str(e))
            raise e
        
        #logging.info("Epoch Train Loss: %.4f, Mean Anomaly Macro F1: %.4f, Mean Macro F1: %.4f, Mean Balanced Acc: %.4f",
        #             mean_loss, mean_anomaly_macro_f1, mean_macro_f1, mean_balanced_acc)
    def get_label_metrics(self, y_true, y_pred, mean_loss, export_results: bool = False, is_final_test: bool = False):
        """Calculate classification metrics for labels."""        

        precision_val = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_val = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_score_val = f1_score(y_true, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        # f1 score for anomaly classes only (1..5)
        y_true_bin = [1 if label > 0 else 0 for label in y_true]
        y_pred_bin = [1 if label > 0 else 0 for label in y_pred]
        anomaly_f1_score = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=1,
            average="binary",
            zero_division=0,
        )

        # f1 score normal vs anomaly
        binary_f1_score = f1_score(
            y_true_bin,
            y_pred_bin,
            average="macro",
            zero_division=0
        )

        if export_results:
            with open(f"./exports/results/{'final_' if is_final_test else ''}gnn_het_multi_classification_perf_scores.csv", "w") as f:
                f.write("Metric,Value\n")
                f.write(f"loss,{mean_loss}\n")
                f.write(f"precision,{precision_val}\n")
                f.write(f"recall,{recall_val}\n")
                f.write(f"f1_score,{f1_score_val}\n")
                f.write(f"accuracy,{accuracy}\n")
                f.write(f"balanced_accuracy,{balanced_accuracy}\n")
                f.write(f"anomaly_f1_score,{anomaly_f1_score}\n")
                f.write(f"binary_f1_score,{binary_f1_score}\n")

        

        return {
            "loss": mean_loss,
            "precision": precision_val,
            "recall": recall_val,
            "f1_score": f1_score_val,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "anomaly_f1_score": anomaly_f1_score,
            "binary_f1_score": binary_f1_score,
            "early_stop_metric": (anomaly_f1_score + f1_score_val) / 2.0
        } 

    @torch.no_grad()
    def evaluate_model(self, loader: DataLoader):
        self.eval()
        
        total_loss = 0
        total_num = 0
        y_all = []
        y_pred_all = []
        for batch in loader:
            batch = batch.to(DEVICE)
            anom_logits = self(batch)      # [B,5]
            y = batch.y.view(-1).long()
            loss = self.criterion(anom_logits, y)
            
            total_loss += loss.item() * y.size(0)
            total_num += y.size(0)
            pred = anom_logits.argmax(dim=1)
            y_all.extend(y.detach().cpu().tolist())
            y_pred_all.extend(pred.detach().cpu().tolist())
        mean_loss = total_loss / max(1, total_num)
        return self.get_label_metrics(y_all, y_pred_all, mean_loss)

    @torch.no_grad()
    def predict(self, data: HeteroData) -> List[int]:
        """Predict anomaly classes for the given data HeteroData."""
        self.eval()
        data = data.to(DEVICE)
        batch = Batch.from_data_list([data]).to(DEVICE)
        anom_logits = self(batch)      # [B,5]
        pred = anom_logits.argmax(dim=1)
        return pred.detach().cpu().tolist()

    def fit_model(self, 
                train_loader: DataLoader,  
                val_loader: DataLoader,
                config: Dict[str, Any]):
        """Train the GNN model with early stopping based on validation anomaly macro F1 score."""
        num_epochs = config.get("max_epochs", 100)
        learning_rate = config.get("learning_rate", 0.001)
        patience = config.get("early_stopping_patience", 10)
        min_delta = config.get("early_stopping_min_delta", 0.0001)
        weight_decay = config.get("weight_decay", 1e-5)

        criterion = self.get_criterion(train_loader)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10,min_lr=1e-6)
        early_stop_mode = 'min'  # We want to maximize F1 score
        early_stopper = GNNEarlyStopping(patience=patience, min_delta=min_delta, mode=early_stop_mode)
        best_val_metrics = None
        best_model_state = None
        early_stop_metric = 'loss'
        
        

        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, optimizer)
            val_metrics = self.evaluate_model(val_loader)
            logging.info("Stage 2 Epoch %d: stop_score: %.4f Train Loss: %.4f, Val Loss: %.4f, f1 Score: %.4f, anomaly f1 Score: %.4f, binary f1 Score: %.4f",
                        epoch+1, val_metrics[early_stop_metric], train_metrics["loss"], val_metrics["loss"], val_metrics["f1_score"], val_metrics["anomaly_f1_score"], val_metrics["binary_f1_score"])
            
            scheduler.step(val_metrics[early_stop_metric])
            # Check for early stopping
            if best_val_metrics is None or (early_stop_mode == 'max' and val_metrics[early_stop_metric] > best_val_metrics[early_stop_metric]) or (early_stop_mode == 'min' and val_metrics[early_stop_metric] < best_val_metrics[early_stop_metric]):
                best_val_metrics = val_metrics
                best_model_state = self.state_dict()
                logging.info("New best model found at epoch %d with F1 Score: %.4f", epoch, val_metrics[early_stop_metric])

            early_stopper.step(val_metrics[early_stop_metric])
            if early_stopper.early_stop:
                logging.info("Early stopping triggered at epoch %d during Stage 2.", epoch+1)
                break
        # Load best model state
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        # Calculate final training metrics
        final_train_metrics = self.evaluate_model(train_loader)
        logging.info("Final Training Metrics - Loss: %.4f, F1: %.4f, Recall: %.4f, Precision: %.4f, Balanced Acc: %.4f, Accuracy: %.4f",
                    final_train_metrics["loss"], final_train_metrics["f1_score"], final_train_metrics["recall"],
                    final_train_metrics["precision"], final_train_metrics["balanced_accuracy"], final_train_metrics["accuracy"])

        return self, criterion, final_train_metrics

    def test_model(self, test_loader: DataLoader, test_description: str="Final Test Set"):
        """Evaluate the trained model on the final test set and print classification report."""
        #criterion = self.get_criterion(test_loader)
        test_metrics = self.evaluate_model(test_loader)
        logging.info("%s Results - Loss: %.4f, F1: %.4f, Recall: %.4f, Precision: %.4f, Balanced Acc: %.4f, Accuracy: %.4f",
                    test_description, test_metrics["loss"], test_metrics["f1_score"], test_metrics["recall"],
                    test_metrics["precision"], test_metrics["balanced_accuracy"], test_metrics["accuracy"])
        
        # Get detailed classification report
        y_all = []
        y_pred_all = []
        total_loss = 0
        total_num = 0
        for batch in test_loader:
            batch = batch.to(DEVICE)
            anom_logits = self(batch)      # [B], [B,5]
            y = batch.y.view(-1).long()

            #loss = self.criterion(anom_logits, y)
            loss = self.criterion(anom_logits, y)
            total_loss += loss.item() * y.size(0)
            total_num += y.size(0)
            pred = anom_logits.argmax(dim=1)
            y_all.extend(y.detach().cpu().tolist())
            y_pred_all.extend(pred.detach().cpu().tolist())
        
        

        self.get_label_metrics(y_all, y_pred_all, total_loss / max(1, total_num), export_results=True)
        report = classification_report(y_all, y_pred_all, target_names=y_labels, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"./exports/results/classification_report_classify_{test_description}.csv")

                # Confusion Matrix
        
        # save confusion matrix image
        cm = confusion_matrix(y_all, y_pred_all)
        # add timestamp to filename to avoid overwriting
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"./exports/images/gnn_het_classification_classify_confusion_matrix{int(time.time())}.png")
        plt.close()

        #Get Explainability for all anomaly predictions
        for i in range(len(y_pred_all)):
            if y_pred_all[i] > 0:
                logging.info("Generating explanation for test sample %d with predicted class %d (%s)", i, y_pred_all[i], y_labels[y_pred_all[i]])
                data = test_loader.dataset[i]
                explainer_results = self.explain_with_captum(data)
                #Save or log explainer_results as needed
        

        return y_all, y_pred_all

    # Helper functions for training and evaluation
    def get_weights(self, labels, min_num_classes, epsilon=1e-6):
        """Compute class weights to handle class imbalance."""
        counts = np.bincount(labels, minlength=min_num_classes).astype(np.float32)
        counts[counts == 0] = epsilon  # avoid division by zero
        weights = 1.0 / counts
        weights = weights / np.sum(weights) * len(counts)  # normalize
        weights[~np.isfinite(weights)] = epsilon  # handle any inf or nan
        
        
        
        return weights


    def explain_with_captum(self, data: HeteroData, save_dir="./exports/explanations"):
        """
        Generates explanations with Snapshot ID, True/Pred Y, and Global Feature Plots.
        """
        self.eval()
        os.makedirs(save_dir, exist_ok=True)
        timestamp = int(time.time())

        # Extract Metadata (Snapshot, True Y, etc.)
        # Handle Snapshot ID (support string, int, or tensor)
        snapshot_id = "unknown"
        if hasattr(data, 'snapshot_id'):
            s_id = data.snapshot_id
            if torch.is_tensor(s_id):
                snapshot_id = str(s_id.item()) if s_id.numel() == 1 else str(s_id.tolist())
            else:
                snapshot_id = str(s_id)
        
        # Handle True Label
        true_label_idx = data.y.item() if hasattr(data, 'y') and data.y.numel() == 1 else -1
        true_label_name = y_labels[true_label_idx] if 0 <= true_label_idx < len(y_labels) else "Unknown"

        # Get Model Prediction
        data = data.to(DEVICE)
        batch = Batch.from_data_list([data]).to(DEVICE)
        
        with torch.no_grad():
            logits = self(batch)
            pred_class = logits.argmax(dim=1).item()
            pred_prob = F.softmax(logits, dim=1).max().item()
            pred_label_name = y_labels[pred_class]

        # Filter: Only explain anomalies (Optional: remove if you want to explain everything)
        if pred_class == 0:
            logging.info(f"Skipping explanation for Normal traffic (Snapshot: {snapshot_id}).")
            return None

        logging.info(f"Explaining Snapshot {snapshot_id}: True: {true_label_name} -> Pred: {pred_label_name} ({pred_prob:.4f})")

        # Prepare Inputs for Captum
        inputs_list = []
        node_types = []
        for ntype in data.x_dict.keys():
            if data[ntype].num_nodes > 0 and hasattr(data[ntype], "x"):
                inputs_list.append(data[ntype].x.clone().detach().requires_grad_(True))
                node_types.append(ntype)

        inputs_tuple = tuple(inputs_list)
        baselines_tuple = tuple(torch.zeros_like(t) for t in inputs_tuple)

        # Run Attribution
        # Uses the static wrapper defined previously
        forward_func = partial(_fast_model_forward_wrapper, self, data, DEVICE)
        ig = IntegratedGradients(forward_func=forward_func)

        try:
            attributions = ig.attribute(
                inputs=inputs_tuple,
                baselines=baselines_tuple,
                target=pred_class,
                n_steps=50,
                internal_batch_size=10
            )
        except Exception as e:
            logging.error(f"Error during IG attribution for Snapshot {snapshot_id}: {e}")
            return None

        # Process Results ---
        explanation_data = {
            "meta": {
                "timestamp": timestamp,
                "snapshot_id": snapshot_id,
                "true_y": true_label_idx,
                "true_label": true_label_name,
                "predicted_y": pred_class,
                "predicted_label": pred_label_name,
                "confidence": pred_prob
            },
            "node_importances": [],
            "feature_importances": {}
        }
        
        all_node_rankings = []
        global_feature_records = [] # For the Bar Chart

        for idx, ntype in enumerate(node_types):
            attr_tensor = attributions[idx].detach().cpu()
            
            # Feature Importance
            feat_imp = attr_tensor.abs().mean(dim=0).numpy()
            num_features = len(feat_imp)
            
            # Safe Name Mapping
            col_names = []
            try:
                col_name_key = ntype
                if ntype in ["TankMeasurements", "ValveMeasurements"]:
                    col_name_key = "Measurements"
                elif ntype in ["PumpMeasurements", "SensorMeasurements"]:
                    col_name_key = "StateMeasurements"
                else:
                    if not ntype.endswith("s"):
                        col_name_key = ntype + "s"
                col_names = get_hetero_column_names(col_name_key)
            except Exception:
                pass

            if len(col_names) != num_features:
                col_names = [f"feat_{i}" for i in range(num_features)]

            # Store in JSON
            explanation_data["feature_importances"][ntype] = {
                k: float(v) for k, v in zip(col_names, feat_imp)
            }

            # Collect for Global Plot
            for fname, fscore in zip(col_names, feat_imp):
                global_feature_records.append({
                    "node_type": ntype,
                    "feature": fname,
                    "full_name": f"{ntype}: {fname}",
                    "score": float(fscore)
                })

            # Node Importance & Mapping
            node_imp = attr_tensor.abs().sum(dim=1).numpy()
            num_nodes = len(node_imp)
            
            # Safe ID Mapping
            orig_ids = []
            if hasattr(data[ntype], 'original_id'):
                raw = data[ntype].original_id
                orig_ids = raw.tolist() if torch.is_tensor(raw) else raw
            elif hasattr(data[ntype], 'n_id'):
                orig_ids = data[ntype].n_id.tolist()
            
            if len(orig_ids) != num_nodes:
                orig_ids = [f"{ntype}_{i}" for i in range(num_nodes)]

            # Create Records
            for i, score in enumerate(node_imp):
                if score > 1e-4:
                    record = {
                        "snapshot_id": snapshot_id, # Added to record
                        "node_type": ntype,
                        "pyg_index": i,
                        "original_id": str(orig_ids[i]),
                        "importance_score": float(score),
                        "true_label": true_label_name,
                        "pred_label": pred_label_name
                    }
                    explanation_data["node_importances"].append(record)
                    all_node_rankings.append(record)

        # Generate Global Top 5 Feature Plot
        if global_feature_records:
            # Sort by score descending
            global_feature_records.sort(key=lambda x: x['score'], reverse=True)
            top_5_features = global_feature_records[:5]
            
            # Extract data for plotting
            plot_names = [x['full_name'] for x in top_5_features]
            plot_scores = [x['score'] for x in top_5_features]
            
            # Plotting
            plt.figure(figsize=(10, 6))
            sns.barplot(x=plot_scores, y=plot_names, palette="viridis")
            plt.title(f"Top 5 Features (Snapshot {snapshot_id})\nTrue: {true_label_name} | Pred: {pred_label_name}")
            plt.xlabel("Mean Absolute Attribution")
            plt.tight_layout()
            
            # Save Plot
            plot_path = f"{save_dir}/plot_top5_feats_snap{snapshot_id}_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close() 
            
            # Add top 5 to explanation data for easy access
            explanation_data["top_5_global_features"] = top_5_features

        # Export Files
        all_node_rankings.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Save JSON
        json_path = f"{save_dir}/explanation_snap{snapshot_id}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(explanation_data, f, indent=2)
            
        # Save CSV
        node_df = pd.DataFrame(all_node_rankings)
        if not node_df.empty:
            node_csv_path = f"{save_dir}/ranking_snap{snapshot_id}_{timestamp}.csv"
            node_df.head(50).to_csv(node_csv_path, index=False)
            
        logging.info(f"Saved explanations and plot for snapshot {snapshot_id} to {save_dir}")
        return explanation_data       

class GNNEarlyStopping:
    """Early stopping utility to stop training when 
    macro F1 score across all anomaly classes does not improve.
    We ignore the normal class (class 0) for early stopping as it is over-represented.    
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
    
    def step(self, val: float):
        if self.best_score is None:
            self.best_score = val
        elif self.mode == 'max' and val > self.best_score + self.min_delta:
            self.best_score = val
            self.counter = 0
        elif self.mode == 'min' and val < self.best_score - self.min_delta:
            self.best_score = val
            self.counter = 0
        # elif val == 0.0:
        #     # special case to avoid early stopping at beginning
        #     pass
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                    
        return self.early_stop

# -------- Explanation functions --------
