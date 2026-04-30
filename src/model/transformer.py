import torch
import torch.nn as nn

from norm import RMSNorm
from attention import LLamaAttension

class Feedforward(nn.Module):
    def __init__(self, dim:int, hidden_dim:int):
        #swiGLU
        self.w1 = nn.Linear(dim,hidden_dim,bias=False)
        self.w2 = nn.Linear(dim,hidden_dim,bias=False)
        self.w3 = nn.Linear(dim,hidden_dim,bias=False)
    def forward(self,x):
        # SwiGLU 公式: w2( SiLU(w1(x)) * w3(x) )
        return self.w2(F.silu(self.w1(x))*self.w3(x))

class LlamaBlock(nn.Module):
    def __init__(self,dim:int, n_heads:int , hidden_dim:int):
        super().__init__()
        self.attention = LLamaAttension(dim, n_heads)
        self.feed_forward = Feedforward(dim, hidden_dim)
        
        # 两个 RMSNorm，分别用于 Attention 前和 FFN 前
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self,x:torch.Tensor,freqs_cis: torch.Tensor, mask: torch.Tensor = None):
        h = x + self.attention(self.attention_norm(x),freqs_cis,mask)
        out = h+self.feed_forward(self.ffn_norm(h))
        return out
    

