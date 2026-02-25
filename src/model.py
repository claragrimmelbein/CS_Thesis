import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

class SimpleTransformerPredictor(nn.Module):
    def __init__(self, input_dim, model_dim):
        super().__init__()
        self.q_linear = nn.Linear(input_dim, model_dim)
        self.k_linear = nn.Linear(input_dim, model_dim)
        self.v_linear = nn.Linear(input_dim, model_dim)
        self.attention = ScaledDotProductAttention(model_dim)
        self.out = nn.Linear(model_dim, 1)

    def forward(self, x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        attn_output = self.attention(Q, K, V)
        return self.out(attn_output.mean(dim=1))