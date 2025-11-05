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
        # self.learnable_tokens = nn.Parameter(torch.zeros(token_num, dim), requires_grad=True)
        self.learnable_tokens = nn.Embedding(token_num, dim)
        nn.init.normal_(self.learnable_tokens, std=0.02)
        # nn.init.xavier_uniform_(self.learnable_tokens)
        
    def forward(self, input, attention_mask=None):
        
        # if not torch.isfinite(self.learnable_tokens).all():
        #     with torch.no_grad():
        #         self.learnable_tokens.data = torch.nan_to_num(
        #             self.learnable_tokens.data, nan=0.0, posinf=1e4, neginf=-1e4
        #         )
        
        # is_nan = torch.isnan(self.learnable_tokens.weight)
        # is_inf = torch.isinf(self.learnable_tokens.weight)  # +inf, -inf 모두 True
        # print(self.learnable_tokens.weight.shape)
        # # 개수 세기
        # num_nan = is_nan.sum()
        # num_inf = is_inf.sum()

        # print(f"NaN 개수: {num_nan}")
        # print(f"Inf 개수: {num_inf}")
        
        # import pdb;pdb.set_trace()
        
        # print(self.learnable_tokens)
        
        k = self.learnable_tokens.weight.to(input.device)
        k = k.unsqueeze(0).transpose(2, 1)
        q = input.unsqueeze(0)

        # print("q.isfinite(): ", q.isfinite().all())
        # print("q.dtype: ", q.dtype)
        # print("k.isfinite(): ", k.isfinite().all())
        # print("k.dtype: ", k.dtype)
        # import pdb;pdb.set_trace()
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
