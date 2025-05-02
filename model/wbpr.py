import torch
import torch.nn as nn

class WBPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(WBPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score, pos_weights, neg_weights):
        # Positive loss: weighted log(sigmoid(pos_score - neg_score))
        pos_loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)) * pos_weights

        # Negative loss: weighted log(sigmoid(neg_score - pos_score))
        neg_loss = -torch.log(self.gamma + torch.sigmoid(neg_score - pos_score)) * neg_weights

        # Weighted sum of positive and negative losses
        total_loss = (pos_loss.sum() + neg_loss.sum()) / (pos_weights.sum() + neg_weights.sum())
        return total_loss