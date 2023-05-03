import torch.nn as nn
import torch

class DistLoss(nn.Module):
    def __init__(self,balance=0.5) -> None:
        super().__init__()
        self.pred_loss_fn = nn.CrossEntropyLoss()
        self.feat_loss_fn = nn.CosineEmbeddingLoss()
        self.balance = balance

    def forward(self,out_pred,out_feat,target_pred,target_feat):
        pred_loss = self.pred_loss_fn(out_pred,target_pred)
        if self.balance < 1.0:
            feat_loss = self.feat_loss_fn(input1=out_feat,input2=target_feat,target=torch.ones(out_feat.shape[0]).cuda())
            dist_loss = self.balance * pred_loss + (1 - self.balance) * feat_loss # <<BALANCE --> <<INFLUENCE OF PREDICTION LOSS
        else:
            dist_loss = pred_loss
            feat_loss = torch.tensor(0.0)
        return pred_loss, feat_loss, dist_loss