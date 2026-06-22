from captum_explainer import CaptumExplainer

class ExplanationReportGenerator:

    def __init__(self):
        pass
      
    def generate_report(
        self,
        data,
        explanation,
        save_dir="./exports/explanations"
    ):
  
    def generate_json(
        self,
        meta,
        node_importances,
        feature_importances,
    ):
        return {
            "meta": meta,
            "node_importances": node_importances,
            "feature_importances": feature_importances,
        }
  
    def save_json(
        self,
        explanation_data,
        path,
    ):
        with open(path, "w") as f:
            json.dump(
                explanation_data,
                f,
                indent=2
            )
  
    def save_csv(
        self,
        node_rankings,
        path,
    ):
        df = pd.DataFrame(node_rankings)
    
        if not df.empty:
            df.to_csv(
                path,
                index=False
            )
  
    def save_node_rankings(
        self,
        rankings,
        path,
    ):
        rankings.sort(
            key=lambda x: x["importance_score"],
            reverse=True,
        )
    
        self.save_csv(rankings, path)
  
    def save_feature_plot(
        self,
        features,
        snapshot_id,
        true_label,
        pred_label,
        path,
    ):
        if len(features) == 0:
            return
    
        top5 = sorted(
            features,
            key=lambda x: x["score"],
            reverse=True,
        )[:5]
    
        names = [x["full_name"] for x in top5]
        scores = [x["score"] for x in top5]
    
        plt.figure(figsize=(10, 6))
    
        sns.barplot(
            x=scores,
            y=names,
            palette="viridis"
        )
    
        plt.title(
            f"Top 5 Features (Snapshot {snapshot_id})\n"
            f"True: {true_label} | Pred: {pred_label}"
        )
    
        plt.xlabel("Mean Absolute Attribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
  
    def save_node_plot(
        self,
        node_rankings,
        path,
    ):
        top10 = sorted(
            node_rankings,
            key=lambda x: x["importance_score"],
            reverse=True,
        )[:10]
    
        names = [
            f"{x['node_type']}:{x['original_id']}"
            for x in top10
        ]
    
        scores = [
            x["importance_score"]
            for x in top10
        ]
    
        plt.figure(figsize=(10, 6))
    
        sns.barplot(
            x=scores,
            y=names,
            palette="magma"
        )
    
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
