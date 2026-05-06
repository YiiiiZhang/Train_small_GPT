import torch
import torch.nn as nn
import torch.nn.Functional as F

def precompute_freqs_cis(dim:int, max_seq_len: int, theta:float= 1000.0):
    freqs = 1.0/(theta ** (torch.arange(0,dim,2)[:(dim // 2)].float()/dim))
    t = torch.arange(max_seq_len, device = freqs.device, dtype = torch.float32)
    freqs = torch.outer(t,freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs),freqs)
    return freqs_cis

def apply_rotary_emb(xq:torch.Tensor,xk:torch.Tensor,freqs_cis:torch.Tensor):
    xq_complex = torch.view_as_complex(xq.folat().reshape(*xq.shape[:-1],-1,2))
    xk_complex = torch.view_as_complex(xk.folat().reshape(*xk.shape[:-1],-1,2))

    freqs_cis = freqs_cis.view(1,xq_complex.shape[1],1,xq_complex.shape[-1])
    xq_out = torch.view_as_real(xq_complex*freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex*freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class LLamaAttension(nn.Module):
    def __init__(self,dim:int,n_heads:int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim//n_heads
        assert self.head_dim * n_heads == dim, "dim 必须能被 n_heads 整除" 
        # dim*dim * 4
        self.wq = nn.Linear(dim,n_heads*self.head_dim,bias=False)
        self.wk = nn.Linear(dim,n_heads*self.head_dim,bias=False)
        self.wv = nn.Linear(dim,n_heads*self.head_dim,bias=False)
        self.wo = nn.Linear(n_heads*self.head_dim,dim,bias=False)
    
    def forward(self,x:torch.Tensor,freqs_cis:torch.Tensor,mask: torch.Tensor = None):
        B,T,C = x.shape

        q = self.wq(x).view(B,T,self.n_heads,self.head_dim)
        k = self.wk(x).view(B,T,self.n_heads,self.head_dim)
        v = self.wv(x).view(B,T,self.n_heads,self.head_dim)

        q,k = apply_rotary_emb(q,k,freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q,k,v,
            attn_mask = mask,
            dropout = 0.0,
            is_causal=True if mask is None else False
        )
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(output)