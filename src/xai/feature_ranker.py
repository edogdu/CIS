# feat_imp = attr_tensor.abs().mean(dim=0)

class FeatureRanker:

    @staticmethod
    def rank(
        attr_tensor,
        feature_names,
    ):
        
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









# input extraction
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
