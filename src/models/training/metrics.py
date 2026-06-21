from models.encoders.hetero_encoder import (
    GNNEvaluator
)
        
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
