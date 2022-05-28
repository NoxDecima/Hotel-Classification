import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcFace(nn.Module):
    def __init__(self, in_features: int, out_features: int, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor = None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if self.training:
            cosine.acos_()

            index = torch.where(label != -1)[0]
            m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
            m_hot.scatter_(1, label[index, None], self.m)
            cosine[index] += m_hot

            cosine.cos_()

        cosine.mul_(self.s)

        return cosine
