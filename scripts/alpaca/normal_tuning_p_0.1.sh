#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# bash scripts/alpaca/normal_tuning_p_0.1.sh> finetuned_logs/alpaca/normal_tuning_p_0.1.log 2>&1 &
echo $CUDA_VISIBLE_DEVICES

torchrun --nnodes 1 --nproc_per_node 1  --master_port 29504 finetuning.py \
--batch_size_training 8 --lr 5e-5 \
--num_epochs 10 \
--dataset alpaca_dataset \
--mode 1k_p_0.1 \
--model_name ckpts/Llama-2-7b-chat-fp16 \
--pure_bf16 \
--dist_checkpoint_root_folder finetuned_models \
--dist_checkpoint_folder alpaca-7b-full \
--output_dir finetuned_models/alpaca/normal_tuning_p_0.1 \
--use_peft True \
--lambda_reg 1 \
--gradient_accumulation_steps 1 --run_validation False --save_every_epoch False

# --enable_fsdp \