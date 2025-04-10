import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_item_popularity(item_num, item_id):
    """
    Count the number of times each id appears in the item_id tensor
    Args:
        - item_num (int)
        - item_id (torch.Tensor)
    """
    counts = torch.bincount(item_id, minlength=item_num)
    return {item_id: count.item() for item_id, count in enumerate(counts)}

def calculate_weight(item_popularity, alpha, beta):
    """
    w_F(v) = alpha - tanh(beta*F(v) - beta)
    """
    weight = alpha - torch.tanh((beta * item_popularity - beta).clone().detach())
    return weight

def WCE(logits, target, weights):
    log_probs = F.log_softmax(logits, dim=1)
    criterion = nn.NLLLoss(reduction='none')
    loss = criterion(log_probs, target)
    loss = loss * weights
    return loss.sum() / weights.sum()