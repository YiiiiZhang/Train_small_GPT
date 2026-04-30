import torch
import torch.nn as nn
import torch.nn.functional as F

from norm import RMSNorm
from attention import precompute_freqs_cis
from transformer import LlamaBlock

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
