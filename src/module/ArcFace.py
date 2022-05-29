import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.2):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m * math.pi

    def forward(self, cosine: torch.Tensor, label: torch.Tensor):
        cosine.acos_()

        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] += m_hot

        cosine.cos_().mul_(self.s)

        return cosine


class NormalizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(NormalizedLinear, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor):
        return F.linear(F.normalize(x), F.normalize(self.weight), bias=None)
