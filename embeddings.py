import torch
import numpy as np
import torch.nn as nn
import math

__all__ = ["SinusoidalPositionEmbeddings", "LearnablePositionalEmbedding2D", "PositionalEmbedding2D"]


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PointEmbedder(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=16):
        super(PointEmbedder, self).__init__()
        # Linear layer to embed the 2D points into a higher-dimensional space
        self.fc = nn.Linear(input_dim, embedding_dim)
        
    def forward(self, x):
        # x has shape [batch_size, num_points, 2]
        # Pass through the linear layer
        x = self.fc(x)  # Now x has shape [batch_size, num_points, embedding_dim]
        return x
