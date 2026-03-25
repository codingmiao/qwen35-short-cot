#!/bin/bash

# 本脚本用到的包
# vllm 0.15.1
# ms_swift 4.0.1
# qwen-vl-utils 0.0.14 . 可选，多模态微调时用到
# deepspeed 0.18.8 . 可选，显存不足时，安装deepspeed 并使用 --deepspeed zero3 甚至 --deepspeed zero3_offload 来分配显存甚至分配到内存

# ==================== 1. 基础配置参数 ====================
MODEL_PATH="/mydata/models/hf/0225/Qwen3.5-9B/"
DATASET_PATH="/mydata/models/sft/0313/filtered_r1_messages.jsonl"
OUTPUT_DIR="/mydata/models/sft/0313/output"

# ==================== 2. GPU 与 分布式配置 ====================
GPU_IDS="0,1,2,3"
GPUS_COUNT=4
MAX_MEM_STR='{0: "64GB", 1: "64GB", 2: "60GB", 3: "64GB"}'

# ==================== 3. 训练超参数 ====================
BATCH_SIZE=4
ACCUMULATION=4
LR="5e-5"
EPOCHS=2
MAX_LEN=2048
LORA_RANK=8
LORA_ALPHA=32

# ==================== 4. 启动训练 ====================

echo "使用 GPU: $GPU_IDS 进行 Swift 多卡 SFT（不使用 torchrun）"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export HIP_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mydata/models/sft/0313/

# Swift 负责自动 fork 多个进程
NPROC_PER_NODE=$GPUS_COUNT swift sft \
    --model "$MODEL_PATH" \
    --train_type lora \
    --dataset "$DATASET_PATH" \
    --torch_dtype bfloat16 \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 10 \
    --learning_rate $LR \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --target_modules all-linear \
    --gradient_accumulation_steps $ACCUMULATION \
    --save_steps 5 \
    --save_total_limit 10 \
    --report_to tensorboard \
    --logging_steps 1 \
    --max_length $MAX_LEN \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing true \
    --loss_scale ignore_empty_think \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --max_memory "$MAX_MEM_STR"
