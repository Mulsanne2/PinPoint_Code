PROMPT_VERSION="vicuna_v1"
RUN_NAME="llava-1.6-vicuna7b-lora_e10"
PREV_STAGE_CHECKPOINT="liuhaotian/llava-v1.6-vicuna-7b"
# --deepspeed ./scripts/zero3.json \
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 4 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/dataset_final/info/pinpoint_info_train_val_format.json \
    --image_folder /root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/infographic/images \
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
    --output_dir /root/Desktop/workspace/kwon/pinpoint/LLaVA-NeXT/ckpt/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16