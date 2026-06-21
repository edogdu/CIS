from models.encoders.hetero_encoder import (
    GNNTrainer
)

class GNNEvaluator(nn.Module):

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
