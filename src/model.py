import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        # matrix of [max_len, d_model] representing the position 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape : batch_size, seq_len, d_model
        return x + self.pe[:, :x.size(1)]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        # standard attention mechanism
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

class SimpleTransformerPredictor(nn.Module):
    def __init__(self, input_dim=3, model_dim=64, seq_len=10):
        super().__init__()
        # linear layer
        # to project 3 features: watch, latency, hour to model space
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_len=seq_len)
        
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        
        self.attention = ScaledDotProductAttention(model_dim)
        
        # output layer
        # take attention output and predict a single scalar 
        # this single scalar is the next watch time
        self.fc_out = nn.Sequential(
            nn.Linear(model_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [batch, 10, 3] -> [batch, 10, 64]
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        attn_output = self.attention(Q, K, V)
        
        # avearge across the sequence (10 videos) to get  global representation
        pooled = attn_output.mean(dim=1) 
        
        return self.fc_out(pooled)