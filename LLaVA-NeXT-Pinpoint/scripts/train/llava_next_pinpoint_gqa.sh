PROMPT_VERSION="vicuna_v1"
RUN_NAME="gqa_pinpoint1"
PREV_STAGE_CHECKPOINT="liuhaotian/llava-v1.6-vicuna-7b"

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/dataset_final/gqa/pinpoint_gqa_train_llava.json \
    --image_folder /root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/temp/image/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir /root/Desktop/workspace/kwon/pinpoint/LLaVA-NeXT/ckpt/gqa/$RUN_NAME \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_pinpoint True