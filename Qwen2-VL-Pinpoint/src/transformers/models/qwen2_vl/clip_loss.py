import torch
from torch import nn
import torch.nn.functional as F

def compute_contrastive_loss(image_features, text_features, temperature=0.1):
    # 일반적으로 임베딩은 이미 정규화(normalized)되어있지만, 혹시 모를 경우 정규화합니다.
    image_features = F.normalize(image_features, p=2, dim=-1, eps=1e-6)
    text_features = F.normalize(text_features, p=2, dim=-1, eps=1e-6)
    
    # 유사도 행렬 계산: [B, d] x [d, B] => [B, B]
    logits = image_features @ text_features.T
    logits = logits / temperature  # 온도 스케일링
    
    # 정답 레이블: 각 이미지-텍스트 쌍은 같은 배치 인덱스를 가지므로, 대각선에 해당하는 인덱스가 정답
    batch_size = image_features.size(0)
    labels = torch.arange(batch_size, device=image_features.device)
    
    # 양방향 cross entropy loss 계산
    loss_i2t = F.cross_entropy(logits, labels)      # 이미지->텍스트
    loss_t2i = F.cross_entropy(logits.T, labels)      # 텍스트->이미지
    loss = (loss_i2t + loss_t2i) / 2.0
    
    return loss 

def compute_intra_image_loss(image_features, text_features, temperature=0.1):
    ques_feature_norm = F.normalize(text_features.unsqueeze(0), dim=-1)      # Shape: [1, D]
    image_candidates_norm = F.normalize(image_features, dim=-1) # Shape: [1+Num_neg, D]

    # [1, D] @ [D, 1+Num_neg] => [1, 1+Num_neg]
    logits = torch.matmul(ques_feature_norm, image_candidates_norm.T) / temperature

    # CrossEntropyLoss (answer will be index 0)
    target = torch.zeros(1, dtype=torch.long, device=image_features.device)
    loss = F.cross_entropy(logits, target)
    return loss