import math
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableAttention(nn.Module):
    def __init__(self, token_num, dim):
        super(LearnableAttention, self).__init__()
        self.dim = dim
        self.token_num = token_num
        self.learnable_tokens = nn.Parameter(torch.zeros(token_num, dim), requires_grad=True)
    
        # self.learnable_tokens = nn.Parameter(torch.empty(token_num, dim))
        # nn.init.xavier_uniform_(self.learnable_tokens)
        
    def forward(self, input, attention_mask=None):
        
        k = self.learnable_tokens
        k = k.unsqueeze(0).transpose(2, 1) # (B, C, T)
        q = input.unsqueeze(0) # (B, T, C)
        
        scores = torch.matmul(q, k)
        scores = scores / math.sqrt(self.dim) #scale dot norm
        scores = scores.permute(0, 2, 1)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        att_weight = torch.softmax(scores, dim=-1) # [b, T, sd_dim]
        attn_output = torch.bmm(att_weight, q) #[b, T, sd_dim]
        attn_output = attn_output.squeeze(0) #[T, sd_dim]

        return attn_output

class SpatialMLP(nn.Module):
    def __init__(self, channels, H, W, hidden_ratio=0.5):
        super().__init__()
        self.channels = channels
        self.H = H
        self.W = W
        self.HW = H * W

        # LayerNorm: 배치 크기 변화에도 안정
        self.norm = nn.LayerNorm(channels)

        # 공간(HW) 차원 간 mixing
        hidden_dim = int(self.HW * hidden_ratio)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(self.HW, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.HW)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == self.H and W == self.W, \
            f"Expected spatial size ({self.H},{self.W}), but got ({H},{W})"

        # [B, C, H, W] → [B, HW, C]
        x = x.view(B, C, self.HW).transpose(1, 2)  # [B, HW, C]

        # LayerNorm (channel dimension)
        x = self.norm(x)

        # 공간(HW) mixing: 채널별로 HW 간 Linear 변환
        x = x.transpose(1, 2)                      # [B, C, HW]
        x = self.spatial_mlp(x)                    # [B, C, HW]
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return x