import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


def get_loss(name):
    if name == "cosface":
        return CosFace()
    elif name == "arcface":
        return ArcFace()
    else:
        raise ValueError()


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, in_features: int, out_features: int, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor = None):
        if self.training:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))

            index = torch.where(label != -1)[0]
            m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
            m_hot.scatter_(1, label[index, None], self.m)
            cosine.acos_()
            cosine[index] += m_hot
            cosine.cos_().mul_(self.s)

            output = cosine
        else:
            output = F.linear(F.normalize(input), F.normalize(self.weight))

        return output