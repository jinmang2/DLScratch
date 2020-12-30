import torch
import torch.nn as nn


class Model(nn.Module):
    
    def __init__(self, num_features, hid_dim, Q):
        super().__init__()
        self.layer1 = nn.Linear(num_features, hid_dim, )
        self.layer2 = nn.Linear(hid_dim, Q+2)
        
    def forward(self, x):
        out = torch.tanh(self.layer1(x))
        return torch.tanh(self.layer2(out))
        
        
class BPMLLLoss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

    def forward(self, input, target):
        # https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkde06a.pdf
        # input's shape is (bsz, Q+2)
        # target's shape is (bsz, Q+2)
        # target's unique value is in [-1, 1]
        loss_sum = 0
        for batch_ix, mask in enumerate(target):
            Y = input[batch_ix][mask == 1]
            Y_bar = input[batch_ix][mask != 1]
            k = len(Y) * len(Y_bar)
            term = -torch.cartesian_prod(Y, -Y_bar).sum(dim=-1)
            loss = (1 / k) * torch.sum(torch.exp(term))
            loss_sum += loss
        return loss_sum
