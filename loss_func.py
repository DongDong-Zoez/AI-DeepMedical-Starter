import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Third-party packages

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, device="cuda", max_m=0.5, weight=None, s=30, from_logits=True, *args, **kwargs):
        super(LDAMLoss, self).__init__()
        # 計算每個類別的 m 值
        self.device = device
        self.from_logits = from_logits
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        # 將 m 值縮放到 0 到 max_m 之間
        m_list = m_list * (max_m / np.max(m_list))
        # 將 m_list 轉為 PyTorch 的浮點數張量
        m_list = torch.FloatTensor(m_list).to(device)
        self.m_list = m_list
        # LDAM 的 scaling factor s，需大於 0
        assert s > 0
        self.s = s
        # 損失函數的權重
        self.weight = weight

    def forward(self, x, targets):

        if self.from_logits:
            x = torch.sigmoid(x)

        # 計算 batch 內每個類別的 m 值
        batch_m = torch.matmul(self.m_list[None, :], targets.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        # 計算每個樣本的 x_m 值
        x_m = x - batch_m

        print(x_m, x)

        # 將 output 設為 x_m 或 x，取決於樣本的類別
        weight_vector = torch.mul(targets, x) + torch.mul((1. - targets), x_m)
        print("ok")
        # 計算交叉熵損失
        return F.binary_cross_entropy(self.s*weight_vector, targets, reduction="none", weight=self.weight)

def focal_loss(input_values, gamma):
    """計算 Focal Loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, num_pos=1, num_neg=1, gamma=0., from_logits=True):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.from_logits = from_logits

    def forward(self, inputs, targets):
        """
        計算 Focal Loss
        :param input: 模型預測輸出
        :param target: 真實標籤
        :return: Focal Loss
        """
        if self.from_logits:
            inputs = torch.sigmoid(inputs)
        inputs = inputs.to(dtype=torch.float64)
        targets = targets.to(dtype=torch.float64)
        
        one_weight = (self.num_neg + self.num_pos) / (self.num_pos)
        zero_weight = (self.num_neg + self.num_pos) / (self.num_neg)

        # Apply the weights
        weight_vector = torch.mul(targets, one_weight) + torch.mul((1. - targets), zero_weight)
        weighted_bce = F.binary_cross_entropy(inputs, targets, reduction="none", weight=weight_vector)

        return focal_loss(weighted_bce, self.gamma)
    
class LMFLoss(nn.Module):
    
    def __init__(self, cls_num_list, alpha=0.5, beta=0.5, gamma=2., *args, **kwargs):
        super().__init__()
        
        self.FocalLoss = FocalLoss(gamma=gamma)
        self.LDAMLoss = LDAMLoss(cls_num_list)
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, x, target):
        
        focal_loss = self.FocalLoss(x, target)
        ldam_loss = self.LDAMLoss(x, target)
        
        lmf_loss = self.alpha * focal_loss + self.beta * ldam_loss
        
        return lmf_loss
    
class WeightedBCE(nn.Module):

    def __init__(self, num_pos=0, num_neg=0, from_logits=True):
        super().__init__()
        self.from_logits = from_logits
        self.num_neg = num_neg
        self.num_pos = num_pos

    def forward(self, inputs, targets):
        if self.from_logits:
            inputs = torch.sigmoid(inputs)
        inputs = inputs.to(dtype=torch.float64)
        targets = targets.to(dtype=torch.float64)
        
        one_weight = (self.num_neg + self.num_pos) / (self.num_pos)
        zero_weight = (self.num_neg + self.num_pos) / (self.num_neg)

        # Apply the weights
        weight_vector = torch.mul(targets, one_weight) + torch.mul((1. - targets), zero_weight)
        weighted_bce = F.binary_cross_entropy(inputs, targets, reduction="mean", weight=weight_vector)

        # Return the average loss
        return torch.mean(weighted_bce)
    
def criterion(loss_func, num_cls, num_pos, *args, **kwargs):
    if loss_func == "WeightedBCE":
        return WeightedBCE(num_cls, num_pos)
    elif loss_func == "LMFLoss":
        return LMFLoss(*args, **kwargs)
    elif loss_func == "FocalLoss":
        return FocalLoss(num_cls, num_pos, *args, **kwargs)
    elif loss_func == "BCE":
        return nn.BCEWithLogitsLoss()
    elif loss_func == "NLLLoss":
        return nn.NLLLoss()
    
if __name__ == "__main__":
    a = torch.tensor([[1,0,1,0,1,1,1,0], [1,0,1,1,1,0,0,0]]).to(dtype=torch.float, device="cuda")
    b = torch.randn(2,8).to(dtype=torch.float, device="cuda")
    loss = nn.BCEWithLogitsLoss()
    print(loss(b ,a))