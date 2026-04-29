import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,dim:int, eps:float=1e-6):
        super().__init()
        self.eps = eps
        self.normalized_shape = (dim,)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
    def forward(self,x):
        dim_count = len(self.normalized_shape)
        dims = tuple([-i for i in range(1, dim_count + 1)])
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim = dims, keepdim = True, unbiased = True)
        x_normalized = (x-mean) /torch.sqrt(var+self.eps)
        return self.weight * x_normalized + self.bias


class  RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        super().__init()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self,x):
        # 计算其均方根
        variance = x.pow(2).mean(-1,keepdim = True)
        # 这个均方根对输入向量进行缩放，并乘以一个可学习的权重参数 $\gamma$
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed.to(x.dtype)