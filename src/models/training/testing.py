from models.encoders.hetero_encoder import (
    GNNEvaluator,
    GNNExplainer
)

from xai import captum_explainer

class GNNTester(nn.Module):

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
