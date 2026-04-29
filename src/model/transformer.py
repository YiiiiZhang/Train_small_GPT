import torch
import torch.nn as nn
import torch.nn.functional as F

from norm import RMSNorm
from attention import precompute_freqs_cis,LLamaAttension

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
    
class LlamaModel(nn.Module):
    def __init__(self,vocab_size: int,dim:int,n_layers: int,n_heads:int, hidden_dim: int, max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        #Token embeeding
        self.tok_embeddings = nn.Embedding(vocab_size,dim)

        #duplicate block
        self.layers = [
            LlamaBlock(dim,hidden_dim,hidden_dim) for _ in range(n_layers)
        ]
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim,vocab_size,bias=False)
        freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)
    def forward(self,tokens: torch.Tensor, targets: torch.Tensor = None):
        B,T = tokens.shape
        #calculate embeedings
        h = self.tok_embeddings(tokens)
        #clip RoPE
        freqs_cis = self.freqs_cis[:T]
        
        for layer in self.layers:
            h = layer(h,freqs_cis)
        h = self.norm(h)
        logits = self.output(h)
        loss = None
        if targets is not None:
            # 展平以便 CrossEntropyLoss 处理
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return {"logits":logits,"loss":loss}

