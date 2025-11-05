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
        
        self.proj_k = nn.Linear(self.dim, self.dim)
        self.proj_q = nn.Linear(self.dim, self.dim)
        self.proj_out = nn.Linear(self.dim, self.dim)
        
        # self.learnable_tokens = nn.Parameter(torch.empty(token_num, dim))
        # nn.init.xavier_uniform_(self.learnable_tokens)
        
    def forward(self, input, attention_mask=None):
        
        k = self.learnable_tokens
        k = self.proj_k(k) # B T C
        k = k.unsqueeze(0).transpose(2, 1) # (B, C, T)
        
        q = self.proj_q(input)
        q = q.unsqueeze(0) # (B, T, C)
        
        scores = torch.matmul(q, k)
        scores = scores / math.sqrt(self.dim) #scale dot norm
        scores = scores.permute(0, 2, 1)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        att_weight = torch.softmax(scores, dim=-1) # [b, T, sd_dim]
        attn_output = torch.bmm(att_weight, q) #[b, T, sd_dim]
        attn_output = attn_output.squeeze(0) #[T, sd_dim]
        attn_output = self.proj_out(attn_output)

        return attn_output

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(B, C)
        out = self.mlp(avg_pool) + self.mlp(max_pool)
        out = self.sigmoid(out).view(B, C, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        out = self.sigmoid(self.conv(concat))
        return x * out

class SpatialMLP(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=3):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        #assert H == self.H and W == self.W, \
        #    f"Expected spatial size ({self.H},{self.W}), but got ({H},{W})"

        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        
        return x