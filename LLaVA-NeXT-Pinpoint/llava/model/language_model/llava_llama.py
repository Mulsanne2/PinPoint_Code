#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from torch.nn import CrossEntropyLoss


# , LlamaModel, LlamaForCausalLM, GenerationConfig
# from .modeling_llama import LlamaModel, LlamaForCausalLM
# from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.modeling_llama import LlamaModel, LlamaForCausalLM

from llava.model.learnable_attention import LearnableAttention, SpatialMLP
from llava.model.clip_loss import compute_contrastive_loss, compute_intra_image_loss
from llava.mm_utils import process_images
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

def visualize_grid_and_target(image, grid_shape, window_size, stride, target_window_coords, encompass_bbox, batch_index):

    org_w, org_h = image.size
    
    base_size = 10.0
    
    if org_w >= org_h:
        figsize = (base_size, base_size * org_h / org_w)
    else:
        figsize = (base_size * org_w / org_h, base_size)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    grid_h, grid_w = grid_shape
    x_scale, y_scale = org_w / grid_w, org_h / grid_h

    for i in range(grid_h + 1):
        ax.axhline(i * y_scale, color='black', linewidth=0.5, alpha=0.5)
    for i in range(grid_w + 1):
        ax.axvline(i * x_scale, color='black', linewidth=0.5, alpha=0.5)

    y_start, x_start = target_window_coords
    px_start, py_start = x_start * x_scale, y_start * y_scale
    rect_width, rect_height = window_size * x_scale, window_size * y_scale

    target_rect = patches.Rectangle(
        (px_start, py_start), rect_width, rect_height,
        linewidth=2, edgecolor='lime', facecolor='red', alpha=0.5
    )
    ax.add_patch(target_rect)

    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = encompass_bbox

    encompass_rect = patches.Rectangle(
        (bbox_x1, bbox_y1),
        bbox_x2 - bbox_x1,
        bbox_y2 - bbox_y1,
        linewidth=2,
        edgecolor='yellow',
        facecolor='red',
        alpha=0.2
    )
    ax.add_patch(encompass_rect)

    ax.set_xlim(0, org_w)
    ax.set_ylim(org_h, 0)

    plt.title(f"Visualization for Batch Index: {batch_index}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_inference_results(
    image_org, 
    grid_shape, 
    window_coords, 
    window_size, 
    answer_window_idx, 
    top_k_indices, 
    top_k_scores,
    ground_truth_bbox
):
    """
    추론 결과를 시각화하는 함수 (수정됨).
    - 초록색 테두리: 정답 BBox와 가장 가까운 윈도우 (Ground Truth)
    - 파란색 박스 (투명): 모델이 예측한 Top-K 윈도우들
    - 그리드 선 추가
    """
    org_w, org_h = image_org.size
    grid_h, grid_w = grid_shape

    base_size = 10.0
    if org_w >= org_h:
        figsize = (base_size, base_size * org_h / org_w)
    else:
        figsize = (base_size * org_w / org_h, base_size)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image_org)
    
    # Grid 좌표를 원본 이미지 좌표로 변환하는 스케일
    scale_x = org_w / grid_w
    scale_y = org_h / grid_h

    # Draw grid lines
    for i in range(grid_h + 1):
        ax.axhline(i * scale_y, color='black', linewidth=0.5, alpha=0.5)
    for i in range(grid_w + 1):
        ax.axvline(i * scale_x, color='black', linewidth=0.5, alpha=0.5)

    # 1. 정답 윈도우 (초록색 테두리) 그리기
    if answer_window_idx != -1: # Ensure there is an answer window
        y_start, x_start = window_coords[answer_window_idx]
        rect_gt = patches.Rectangle(
            (x_start * scale_x, y_start * scale_y),
            window_size * scale_x,
            window_size * scale_y,
            linewidth=5,
            edgecolor='red',
            facecolor='none' # No fill
        )
        ax.add_patch(rect_gt)
    if ground_truth_bbox is not None:
        x1, y1, x2, y2 = ground_truth_bbox
        width = x2 - x1
        height = y2 - y1
        rect_gt_actual = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=3,
            edgecolor='pink',
            facecolor='none'
        )
        ax.add_patch(rect_gt_actual)

    # 2. Top-K 예측 윈도우들 (파란색 투명 박스) 그리기
    for i, idx in enumerate(top_k_indices):
        # Skip if it's the answer window and we want to draw it distinctly
        # if idx == answer_window_idx:
            # continue

        y_start, x_start = window_coords[idx]
        
        rect_pred = patches.Rectangle(
            (x_start * scale_x, y_start * scale_y),
            window_size * scale_x,
            window_size * scale_y,
            linewidth=2,
            edgecolor='b',
            facecolor='blue', # Blue fill
            alpha=0.2 # Translucent
        )
        ax.add_patch(rect_pred)

    ax.set_xlim(0, org_w)
    ax.set_ylim(org_h, 0)
    plt.title("Inference Results Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None
    # rope_scaling: Optional[dict] = {}


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)

        # configure default generation settings
        config.model_type = "llava_llama"
        # config.rope_scaling = None

        self.model = LlavaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing

        self.window_size = 10#10#20#15
        self.stride = 7#7#12#10
        # self.top_k = 3
        self.target_ratio = 0.4
        self.region_aggregate = True
        self.ratio_setting = True

        self.embedding_dim = 4096
        self.pinpoint_img_token_proj_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim)
        )

        self.pinpoint_text_token_proj_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim)
        )


        self.clip_loss = compute_contrastive_loss
        self.intra_image_loss = compute_intra_image_loss

        self.pinpoint_learnable_attention = LearnableAttention(100, self.embedding_dim)
        self.pinpoint_temperature_1 = nn.Parameter(torch.tensor([0.07]), requires_grad=True)
        self.pinpoint_temperature_2 = nn.Parameter(torch.tensor([0.07]), requires_grad=True)
        
        self.pinpoint_spatial_mlp = SpatialMLP(self.embedding_dim, 16, 3)

        init_func = nn.init.uniform_
        for module_list in [self.pinpoint_img_token_proj_layer, self.pinpoint_text_token_proj_layer]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        self.post_init()
        self.padding_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        self.bbox_acc = None
        self.tokens_bf = None
        self.tokens_af = None
        self.image_ratio = None

        self.alpha = 0.5

    def get_model(self):
        return self.model

    def _add_noise_and_normalize(self, embedding, sigma=0.004):
        """
        Adds Gaussian noise to an embedding and re-normalizes it.
        Only applied during training.
        """
        if self.training and sigma > 0.0:
            noise = torch.randn_like(embedding) * sigma
            noisy_embedding = embedding + noise
            return F.normalize(noisy_embedding, p=2, dim=-1)
        return embedding

    def _check_overlap(self, box1, box2):
        if box1[2] <= box2[0] or box1[0] >= box2[2]: return False
        if box1[3] <= box2[1] or box1[1] >= box2[3]: return False
        return True

    def train_icra(self, inputs_embeds, ques_embeds, image_orgs, image_org_sizes, grid_shape, answer_bboxes, encompass_bboxes, image_start_indices, image_lengths):
        ques_result = []
        image_result = []

        batch_size = inputs_embeds.shape[0]

        for ques in ques_embeds:
            ques_proj = self.pinpoint_text_token_proj_layer(ques)
            ques_feature = self.pinpoint_learnable_attention(ques_proj).mean(dim=0) #[T, sd_dim]
            ques_result.append(ques_feature)

        total_intra_image_loss = 0.0
        num_valid_samples_for_intra_loss = 0
        
        for i in range(batch_size):
            start_idx = image_start_indices[i]
            length = image_lengths[i]

            end_idx = start_idx + length
            image_tokens = inputs_embeds[i, start_idx:end_idx]
            high_res_tokens = image_tokens[576:] #high_res

            # Selective Text Token Selection Starts
            # with torch.no_grad():
            #     corr_matrix = F.softmax(torch.matmul(ques_embeds[i], image_tokens.T),dim=-1) #[Q, I]
            #     text_score = torch.mean(corr_matrix, dim=1)
            #     threshold = torch.mean(text_score)
            #     text_indices = torch.where(text_score >= threshold)[0]

            # ques_tokens = ques_embeds[i][text_indices]
            # ques_proj = self.pinpoint_text_token_proj_layer(ques_tokens)
            # ques_feature = self.pinpoint_learnable_attention(ques_proj).mean(dim=0)
            # ques_result.append(ques_feature)
            # Selective Text Token Selection Ends

            h,w = grid_shape[i]
            high_res_grid_with_newline = high_res_tokens.view(h, w + 1, self.embedding_dim)
            high_res_grid_wo_newline = high_res_grid_with_newline[:, :w, :]
            restored_high_res_tokens = high_res_grid_wo_newline.reshape(-1, self.embedding_dim)
            img_proj=self.pinpoint_img_token_proj_layer(restored_high_res_tokens).view(h,w,self.embedding_dim)

            padded_h, padded_w = h, w 
            if h < self.window_size or w < self.window_size:
                padded_h, padded_w = max(h, self.window_size), max(w, self.window_size)
                padded_img_proj = self.padding_token.repeat(padded_h, padded_w, 1)
                padded_img_proj[:h, :w, :] = img_proj
                img_proj = padded_img_proj
                
                attention_mask = torch.zeros(padded_h, padded_w, device=self.device)
                attention_mask[:h, :w] = 1
                
            else:
                attention_mask = torch.ones(h, w, device=self.device)

            window_features, window_coords = [], [] # Store (y_start, x_start) for each window

            y_starts = list(range(0, padded_h - self.window_size + 1, self.stride))
            if y_starts[-1] != padded_h - self.window_size:
                y_starts.append(padded_h - self.window_size)
            x_starts = list(range(0, padded_w - self.window_size + 1, self.stride))
            if x_starts[-1] != padded_w - self.window_size:
                x_starts.append(padded_w - self.window_size)

            for y in y_starts:
                for x in x_starts:
                    patch = img_proj[y : y + self.window_size, x : x + self.window_size, :]
                    patch = patch.reshape(-1, self.embedding_dim) # [100, D]
                    
                    patch_mask = attention_mask[y : y + self.window_size, x : x + self.window_size]
                    patch_mask = patch_mask.reshape(-1).bool() 
                    
                    patch = self.pinpoint_spatial_mlp(patch.view(self.window_size, self.window_size, -1).permute(2, 0, 1).unsqueeze(0))[0].permute(1, 2, 0).view(self.window_size * self.window_size, -1)
                    aggregated_patch_feature = self.pinpoint_learnable_attention(patch, attention_mask=patch_mask).mean(dim=0)
                    
                    window_features.append(aggregated_patch_feature)
                    window_coords.append((y, x))
 
            org_w, org_h = image_org_sizes[i]

            bbox = encompass_bboxes[i][0] # [x1, y1, x2, y2]
            center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            grid_center_x, grid_center_y = int((center_x / org_w) * w), int((center_y / org_h) * h)

            min_dist = float('inf')
            # Find the window index that contains the center point
            target_window_idx = -1
            for idx, (y_start, x_start) in enumerate(window_coords):
                window_center_x = x_start + self.window_size / 2
                window_center_y = y_start + self.window_size / 2
                
                # Calculate Euclidian Distance
                dist = math.sqrt((grid_center_x - window_center_x)**2 + (grid_center_y - window_center_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    target_window_idx = idx
            
            # visualize = False
            # if visualize:
            #     visualize_grid_and_target(
            #         image=image_orgs[i],
            #         grid_shape=(h, w),
            #         window_size=self.window_size,
            #         stride=self.stride,
            #         target_window_coords=window_coords[target_window_idx],
            #         encompass_bbox=encompass_bboxes[i][0],
            #         batch_index=i
            #     )

            all_window_features = torch.stack(window_features)
            
            
            # all_window_features = self.spatial_mlp(all_window_features)
            
            
            
            positive_feature = all_window_features[target_window_idx]
            image_result.append(positive_feature) # shape: [D]

            negative_features = []
            grid_encompass_bbox = [(bbox[0] / org_w) * w, (bbox[1] / org_h) * h, (bbox[2] / org_w) * w, (bbox[3] / org_h) * h]
            
            for idx, (y_start, x_start) in enumerate(window_coords):
                if idx == target_window_idx: 
                    continue
                window_bbox = [float(x_start), float(y_start), float(x_start + self.window_size), float(y_start + self.window_size)]
                if not self._check_overlap(window_bbox, grid_encompass_bbox):
                    negative_features.append(all_window_features[idx])

            if negative_features:
                ques_feature = ques_result[i]
                negative_features = torch.stack(negative_features)
                image_candidates = torch.cat([positive_feature.unsqueeze(0), negative_features], dim=0)
                intra_loss = self.intra_image_loss(image_candidates, ques_feature, self.pinpoint_temperature_2)
                total_intra_image_loss += intra_loss
                num_valid_samples_for_intra_loss += 1

        if num_valid_samples_for_intra_loss > 0: #intra image negative sampling
            intra_image_loss = total_intra_image_loss / num_valid_samples_for_intra_loss
        else:
            intra_image_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        ques_result = torch.stack(ques_result).to(self.device) # Shape: [B, D]
        image_result = torch.stack(image_result).to(self.device) # Shape: [B, D]
        pinpoint_loss = self.clip_loss(ques_result, image_result, self.pinpoint_temperature_1)
        loss = pinpoint_loss + (self.alpha * intra_image_loss)

        return loss
    
    def inference_icra(self, inputs_embeds, ques_embeds, image_orgs, image_org_sizes, grid_shape, encompass_bboxes, image_start_indices, image_lengths, attention_mask, position_ids, cache_position):
        self.tokens_bf = inputs_embeds[0].shape[0]
        image_result = []

        ques = ques_embeds[0]
        ques_proj = self.pinpoint_text_token_proj_layer(ques)
        ques_feature = self.pinpoint_learnable_attention(ques_proj).mean(dim=0) #[D]

        start_idx = image_start_indices[0]
        length = image_lengths[0]

        #slice each component
        end_idx = start_idx + length
        text_before = inputs_embeds[0, :start_idx]
        image_tokens = inputs_embeds[0, start_idx:end_idx]
        text_after = inputs_embeds[0, end_idx:]

        low_res_tokens = image_tokens[:576] #low_res
        high_res_tokens = image_tokens[576:] #high_res

        # Selective Token Selection Starts
        # ques = ques_embeds[0]
        # corr_matrix = F.softmax(torch.matmul(ques, image_tokens.T),dim=-1) #[Q, I]
        # text_score = torch.mean(corr_matrix, dim=1) #[Q]
        # threshold = torch.mean(text_score)
        # text_indices = torch.where(text_score >= threshold)[0]
        # ques = ques[text_indices]
        # ques_proj = self.pinpoint_text_token_proj_layer(ques)
        # ques_feature = self.pinpoint_learnable_attention(ques_proj).mean(dim=0) #[D]
        # Selective Token Selection Ends

        h,w = grid_shape[0]
        original_h, original_w = h, w 
        high_res_grid_with_newline = high_res_tokens.view(h, w + 1, self.embedding_dim)
        high_res_grid_wo_newline = high_res_grid_with_newline[:, :w, :]
        restored_high_res_tokens = high_res_grid_wo_newline.reshape(-1, self.embedding_dim)
        img_proj=self.pinpoint_img_token_proj_layer(restored_high_res_tokens).view(h,w,self.embedding_dim)

        padded_h, padded_w = h, w
        if h < self.window_size or w < self.window_size:
            padded_h = max(h, self.window_size)
            padded_w = max(w, self.window_size)

            padded_img_proj = self.padding_token.repeat(padded_h, padded_w, 1)
            padded_img_proj[:h, :w, :] = img_proj

            attention_mask = torch.zeros(padded_h, padded_w, device=self.device)
            attention_mask[:h, :w] = 1
            img_proj = padded_img_proj
        else:
            attention_mask = torch.ones(h, w, device=self.device)

        window_size = self.window_size
        window_features = []
        window_coords = [] # Store (y_start, x_start) for each window

        y_starts = list(range(0, padded_h - window_size + 1, self.stride))
        # If the last window does not align with the bottom edge, add a window that does.
        if y_starts[-1] != padded_h - window_size:
            y_starts.append(padded_h - window_size)

        x_starts = list(range(0, padded_w - window_size + 1, self.stride))
        # If the last window does not align with the right edge, add a window that does.
        if x_starts[-1] != padded_w - window_size:
            x_starts.append(padded_w - window_size)

        for y in y_starts:
            for x in x_starts:
                # Extract 10x10 patch
                patch = img_proj[y : y + window_size, x : x + window_size, :]
                patch = patch.reshape(-1, self.embedding_dim) # [100, D]

                patch_mask = attention_mask[y : y + window_size, x : x + window_size]
                patch_mask = patch_mask.reshape(-1).bool() 
                
                # Aggregate patch info using learnable attention
                # Add a batch dimension for the attention module
                # Aggregate features into a single vector by averaging
                
                patch = self.pinpoint_spatial_mlp(patch.view(self.window_size, self.window_size, -1).permute(2, 0, 1).unsqueeze(0))[0].permute(1, 2, 0).view(self.window_size * self.window_size, -1)
                aggregated_patch_feature = self.pinpoint_learnable_attention(patch, attention_mask=patch_mask).mean(dim=0)

                window_features.append(aggregated_patch_feature)
                window_coords.append((y, x))

        org_w, org_h = image_org_sizes[0]

        bbox = encompass_bboxes[0][0] # [x1, y1, x2, y2]
        answer_center_x = (bbox[0] + bbox[2]) / 2
        answer_center_y = (bbox[1] + bbox[3]) / 2
        # Map center to grid coordinates
        grid_center_x = int((answer_center_x / org_w) * original_w)
        grid_center_y = int((answer_center_y / org_h) * original_h)

        min_dist = float('inf')
        answer_window_idx = -1
        for idx, (y_start, x_start) in enumerate(window_coords):
            window_center_x = x_start + self.window_size / 2
            window_center_y = y_start + self.window_size / 2
            
            # Calculate Euclidian Distance
            dist = math.sqrt((grid_center_x - window_center_x)**2 + (grid_center_y - window_center_y)**2)
            
            if dist < min_dist:
                min_dist = dist
                answer_window_idx = idx

        window_features = torch.stack(window_features)
        ques_feature = F.normalize(ques_feature.unsqueeze(0), dim=-1) # Shape: [1, D]
        window_features = F.normalize(window_features, dim=-1) # Shape: [num_windows, D]
        
        # Calculate Cosine Similarity
        similarity_scores = (ques_feature @ window_features.T).squeeze() # Shape: [num_windows]

        if self.ratio_setting:
            sorted_scores, sorted_indices = torch.sort(similarity_scores, descending=True)
    
            final_top_k_indices = []
            original_area = org_w * org_h
            scale_x, scale_y = org_w / w, org_h / h

            # 2. k=1부터 시작하여, 선택된 윈도우 영역의 비율이 30%가 넘을 때까지 반복
            for k in range(1, len(sorted_indices) + 1):
                current_indices = sorted_indices[:k].tolist()

                # 선택된 윈도우들을 모두 포함하는 Bounding Box 계산
                min_x_grid, min_y_grid = float('inf'), float('inf')
                max_x_grid, max_y_grid = float('-inf'), float('-inf')

                for idx in current_indices:
                    y_start, x_start = window_coords[idx]
                    min_x_grid = min(min_x_grid, x_start)
                    min_y_grid = min(min_y_grid, y_start)
                    max_x_grid = max(max_x_grid, x_start + window_size)
                    max_y_grid = max(max_y_grid, y_start + window_size)
                
                # 그리드 좌표를 실제 픽셀 좌표로 변환
                pixel_x1 = max(0, min_x_grid) * scale_x
                pixel_y1 = max(0, min_y_grid) * scale_y
                pixel_x2 = min(original_w, max_x_grid) * scale_x
                pixel_y2 = min(original_h, max_y_grid) * scale_y
                
                # 현재 영역 비율 계산
                cropped_area = (pixel_x2 - pixel_x1) * (pixel_y2 - pixel_y1)
                current_ratio = cropped_area / original_area
                
                # 3. 목표 비율(30%)을 넘으면, 해당 k개의 윈도우를 최종 선택하고 루프 종료
                if current_ratio >= self.target_ratio:
                    final_top_k_indices = current_indices
                    self.image_ratio = current_ratio
                    break

            top_k_indices = torch.tensor(final_top_k_indices, device=similarity_scores.device)
            top_k_scores = similarity_scores[top_k_indices]
            print("top_k:", len(top_k_indices))
        else:
            if similarity_scores.shape[0] < self.top_k:
                top_k_scores, top_k_indices = torch.topk(similarity_scores, k=similarity_scores.shape[0])
            else:
                top_k_scores, top_k_indices = torch.topk(similarity_scores, k=self.top_k)
        
        # Visualize Dataset
        visualize=False
        if visualize:
            visualize_inference_results(
                image_org=image_orgs[0],
                grid_shape=(padded_h, padded_w),
                window_coords=window_coords,
                window_size=window_size,
                answer_window_idx=answer_window_idx,
                top_k_indices=top_k_indices.tolist(),
                top_k_scores=top_k_scores.tolist(),
                ground_truth_bbox=encompass_bboxes[0][0],
            )

        new_image = []
        final_crop_boxes_pixels = [] # 정확도 계산을 위해 잘라낸 영역의 픽셀 좌표 저장
        scale_x, scale_y = org_w / w, org_h / h

        image_org = image_orgs[0]

        if self.region_aggregate:
            min_x_grid, min_y_grid = float('inf'), float('inf')
            max_x_grid, max_y_grid = float('-inf'), float('-inf')

            for idx in top_k_indices.tolist():
                y_start, x_start = window_coords[idx]
                min_x_grid = min(min_x_grid, x_start)
                min_y_grid = min(min_y_grid, y_start)
                max_x_grid = max(max_x_grid, x_start + window_size)
                max_y_grid = max(max_y_grid, y_start + window_size)

            min_x_grid, min_y_grid = max(0, min_x_grid), max(0, min_y_grid)
            max_x_grid, max_y_grid = min(original_w, max_x_grid), min(original_h, max_y_grid)

            scale_x, scale_y = org_w / w, org_h / h

            pixel_x1,pixel_y1,pixel_x2,pixel_y2 = min_x_grid * scale_x,min_y_grid * scale_y,max_x_grid * scale_x,max_y_grid * scale_y

            image_org = image_orgs[0]
            new_image = [image_org.crop((pixel_x1, pixel_y1, pixel_x2, pixel_y2))] # Rectangle Shape
            cropped_width, cropped_height = pixel_x2 - pixel_x1, pixel_y2 - pixel_y1
            # original_area = org_w * org_h
            # cropped_area = cropped_width * cropped_height
            # self.image_ratio = cropped_area / original_area

            if (pixel_x1 <= answer_center_x <= pixel_x2) and (pixel_y1 <= answer_center_y <= pixel_y2):
                self.bbox_acc = 1
            else:
                self.bbox_acc = 0

            modified_config = copy.deepcopy(self.config)
            modified_config.image_grid_pinpoints = [[336,336],[336,672],[672,336],[1008,336],[336,1008]]
            detailed_image = process_images(new_image, self.image_processor, self.config)
            detailed_image = detailed_image[0, 1:, :, :, :]
            detailed_image = detailed_image.to(dtype=torch.float16, device=inputs_embeds.device) 

            re_encode_image_tokens = self.anyres_encode_image(detailed_image, new_image[0].size)
            re_encode_image_tokens = re_encode_image_tokens.to(dtype=torch.float16, device=inputs_embeds.device)
        
        elif self.region_aggregate == False:
            for idx in top_k_indices.tolist():
                y_start, x_start = window_coords[idx]

                grid_x1, grid_y1 = max(0, x_start), max(0, y_start)
                grid_x2, grid_y2 = min(original_w, x_start + window_size), min(original_h, y_start + window_size)
                
                pixel_x1,pixel_y1 = int(grid_x1 * scale_x),int(grid_y1 * scale_y)
                pixel_x2,pixel_y2 = int(grid_x2 * scale_x),int(grid_y2 * scale_y)

                final_x1,final_y1 = max(0, pixel_x1),max(0, pixel_y1)
                final_x2,final_y2 = min(org_w, pixel_x2),min(org_h, pixel_y2)
                
                cropped_img = image_org.crop((final_x1, final_y1, final_x2, final_y2))
                new_image.append(cropped_img)
                final_crop_boxes_pixels.append((final_x1, final_y1, final_x2, final_y2))

            is_center_in_any_crop = False
            for (x1, y1, x2, y2) in final_crop_boxes_pixels:
                if (x1 <= answer_center_x <= x2) and (y1 <= answer_center_y <= y2):
                    is_center_in_any_crop = True
                    break
            
            if is_center_in_any_crop:
                self.bbox_acc = 1
            else:
                self.bbox_acc = 0
            
            detailed_image = [self.image_processor.preprocess(new_img, return_tensors="pt")["pixel_values"][0] for new_img in new_image] # Square Image
            detailed_image = torch.stack(detailed_image, dim=0)
            detailed_image = detailed_image.to(dtype=torch.float16, device=inputs_embeds.device) 

            re_encode_image_tokens = self.re_encode_image(detailed_image)
            re_encode_image_tokens = re_encode_image_tokens.to(dtype=torch.float16, device=inputs_embeds.device)

        # new_inputs_embeds = torch.cat([text_before,low_res_tokens, re_encode_image_tokens, text_after], dim=0).unsqueeze(0) #final token concatenation
        new_inputs_embeds = torch.cat([text_before, re_encode_image_tokens, text_after], dim=0).unsqueeze(0)
        new_length = new_inputs_embeds.shape[1]
        new_attention_mask = torch.ones((1, new_length), dtype=attention_mask.dtype, device=attention_mask.device)
        new_position_ids = torch.arange(0, new_length, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(0)
        new_cache_position = torch.arange(0, new_length, dtype=cache_position.dtype, device=cache_position.device)
        self.tokens_af = new_length

        return new_inputs_embeds, new_attention_mask, new_position_ids, new_cache_position

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position=None,
        ques_tokens=None,
        image_orgs=None,
        image_org_sizes=None, 
        answer_bboxes=None,
        encompass_bboxes=None,
        grid_shape=None,
        ques_embeds=None,
        image_start_indices=None,
        image_lengths=None,
        train_pinpoint: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None: #Pre-process in here, For Inference Preprocessing is done in generate function, Usingin training only
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, grid_shape, ques_embeds, image_start_indices, image_lengths) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, ques_tokens)
        
        pinpoint_loss=None
        if train_pinpoint and self.training: # Train IRCA
            pinpoint_loss = self.train_icra(inputs_embeds, 
                                            ques_embeds, 
                                            image_orgs, 
                                            image_org_sizes, 
                                            grid_shape,
                                            answer_bboxes, 
                                            encompass_bboxes,
                                            image_start_indices,
                                            image_lengths)
        elif not train_pinpoint and inputs_embeds is not None: # Inference IRCA
            inputs_embeds, attention_mask, position_ids, cache_position = self.inference_icra(inputs_embeds,
                                            ques_embeds, 
                                            image_orgs,
                                            image_org_sizes,
                                            grid_shape,
                                            encompass_bboxes,
                                            image_start_indices,
                                            image_lengths,
                                            attention_mask,
                                            position_ids,
                                            cache_position
                                            )
            

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else: 
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                train_pinpoint=train_pinpoint,
                pinpoint_loss=pinpoint_loss
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        ques_tokens = None,
        image_orgs = None,
        image_org_sizes = None,
        answer_bboxes = None,
        encompass_bboxes = None,
        image_processor = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        grid_shape = None
        self.image_processor = image_processor
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, grid_shape, ques_embeds, image_start_indices, image_lengths) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, ques_tokens=ques_tokens)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, ques_tokens=ques_tokens, image_orgs=image_orgs, image_org_sizes=image_org_sizes, answer_bboxes=answer_bboxes, encompass_bboxes=encompass_bboxes, grid_shape=grid_shape, ques_embeds=ques_embeds, image_start_indices=image_start_indices, image_lengths=image_lengths, **kwargs), self.bbox_acc, self.tokens_bf, self.tokens_af, self.image_ratio

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, ques_tokens=None, image_orgs=None, image_org_sizes=None, answer_bboxes=None, encompass_bboxes=None, grid_shape=None, ques_embeds=None, image_start_indices=None, image_lengths=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        if past_key_values is not None:
            cur_len = past_key_values[0][0].shape[2]
            if cur_len+1 != kwargs['attention_mask'].shape[1]:
                kwargs['attention_mask']=torch.ones((1, cur_len+1), dtype=kwargs['attention_mask'].dtype, device=kwargs['attention_mask'].device)
                kwargs['cache_position']=torch.tensor([cur_len], dtype=kwargs['cache_position'].dtype, device=kwargs['cache_position'].device)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, image_start_indices=image_start_indices, image_lengths=image_lengths, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if ques_tokens is not None:
            inputs["ques_tokens"] = ques_tokens
        if image_orgs is not None:
            inputs["image_orgs"] = image_orgs
        if image_org_sizes is not None:
            inputs["image_org_sizes"] = image_org_sizes
        if answer_bboxes is not None:
            inputs["answer_bboxes"] = answer_bboxes
        if encompass_bboxes is not None:
            inputs["encompass_bboxes"] = encompass_bboxes
        if grid_shape is not None:
            inputs["grid_shape"] = grid_shape
        if ques_embeds is not None:
            inputs["ques_embeds"] = ques_embeds
        if image_start_indices is not None:
            inputs["image_start_indices"] = image_start_indices
        if image_lengths is not None:
            inputs["image_lengths"] = image_lengths
        return inputs


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
