from models.encoders.hetero_encoder import (
    GNNHeteroClassifierModel
)
        
class GNNTrainer:
  
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
