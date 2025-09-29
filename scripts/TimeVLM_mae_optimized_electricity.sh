#!/bin/bash

# Optimized MAE training script for electricity dataset
# Based on analysis of previous experiments, focusing on:
# - Better learning rate scheduling
# - Increased model capacity for complex patterns
# - Improved regularization
# - Memory-efficient batch sizes

echo "Starting Optimized MAE-base training experiments for electricity dataset..."

# Create logs directory
mkdir -p logs

# GPU has 32GB memory, we can use larger models now
# Experiment 1: Enhanced short-term prediction (96 steps)
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity/electricity.csv \
    --model_id electricity_mae_optimized_512_96_exp1 \
    --model TimeVLM \
    --data custom \
    --features M \
    --seq_len 512 \
    --label_len 48 \
    --pred_len 96 \
    --d_model 256 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des "Optimized MAE-base Experiment" \
    --itr 1 \
    --gpu 0 \
    --use_amp \
    --train_epochs 20 \
    --image_size 56 \
    --norm_const 0.4 \
    --periodicity 24 \
    --three_channel_image True \
    --finetune_vlm False \
    --batch_size 24 \
    --learning_rate 0.0005 \
    --num_workers 16 \
    --vlm_type mae \
    --mae_arch mae_base \
    --mae_finetune_type ln \
    --mae_ckpt_dir ./ckpt/ \
    --mae_load_ckpt True \
    --use_mem_gate True \
    --dropout 0.2 \
    --d_ff 1024 \
    --n_heads 8 \
    --percent 1 > logs/TimeVLM_mae_optimized_electricity_512_96_exp1.log 2>&1 &

echo "Running Optimized MAE-base experiment: pred_len=96, d_model=256, experiment_id=exp1"
echo "Log file: logs/TimeVLM_mae_optimized_electricity_512_96_exp1.log"
echo "Experiment started in background with PID: $!"

# Experiment 2: Enhanced medium-term prediction (192 steps)
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity/electricity.csv \
    --model_id electricity_mae_optimized_512_192_exp2 \
    --model TimeVLM \
    --data custom \
    --features M \
    --seq_len 512 \
    --label_len 48 \
    --pred_len 192 \
    --d_model 256 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des "Optimized MAE-base Experiment" \
    --itr 1 \
    --gpu 0 \
    --use_amp \
    --train_epochs 20 \
    --image_size 56 \
    --norm_const 0.4 \
    --periodicity 24 \
    --three_channel_image True \
    --finetune_vlm False \
    --batch_size 24 \
    --learning_rate 0.0005 \
    --num_workers 16 \
    --vlm_type mae \
    --mae_arch mae_base \
    --mae_finetune_type ln \
    --mae_ckpt_dir ./ckpt/ \
    --mae_load_ckpt True \
    --use_mem_gate True \
    --dropout 0.2 \
    --d_ff 1024 \
    --n_heads 8 \
    --percent 1 > logs/TimeVLM_mae_optimized_electricity_512_192_exp2.log 2>&1 &

echo "Running Optimized MAE-base experiment: pred_len=192, d_model=256, experiment_id=exp2"
echo "Log file: logs/TimeVLM_mae_optimized_electricity_512_192_exp2.log"
echo "Experiment started in background with PID: $!"

# Experiment 3: Enhanced long-term prediction (336 steps) with reduced batch size
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity/electricity.csv \
    --model_id electricity_mae_optimized_512_336_exp3 \
    --model TimeVLM \
    --data custom \
    --features M \
    --seq_len 512 \
    --label_len 48 \
    --pred_len 336 \
    --d_model 384 \
    --e_layers 4 \
    --d_layers 2 \
    --factor 5 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des "Optimized MAE-base Experiment" \
    --itr 1 \
    --gpu 0 \
    --use_amp \
    --train_epochs 20 \
    --image_size 56 \
    --norm_const 0.4 \
    --periodicity 24 \
    --three_channel_image True \
    --finetune_vlm False \
    --batch_size 16 \
    --learning_rate 0.0003 \
    --num_workers 16 \
    --vlm_type mae \
    --mae_arch mae_base \
    --mae_finetune_type ln \
    --mae_ckpt_dir ./ckpt/ \
    --mae_load_ckpt True \
    --use_mem_gate True \
    --dropout 0.25 \
    --d_ff 1536 \
    --n_heads 12 \
    --percent 1 > logs/TimeVLM_mae_optimized_electricity_512_336_exp3.log 2>&1 &

echo "Running Optimized MAE-base experiment: pred_len=336, d_model=384, experiment_id=exp3"
echo "Log file: logs/TimeVLM_mae_optimized_electricity_512_336_exp3.log"
echo "Experiment started in background with PID: $!"

# Experiment 4: Very long-term prediction (720 steps) with memory optimization
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity/electricity.csv \
    --model_id electricity_mae_optimized_512_720_exp4 \
    --model TimeVLM \
    --data custom \
    --features M \
    --seq_len 512 \
    --label_len 64 \
    --pred_len 720 \
    --d_model 384 \
    --e_layers 4 \
    --d_layers 2 \
    --factor 5 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des "Optimized MAE-base Experiment" \
    --itr 1 \
    --gpu 0 \
    --use_amp \
    --train_epochs 25 \
    --image_size 56 \
    --norm_const 0.4 \
    --periodicity 24 \
    --three_channel_image True \
    --finetune_vlm False \
    --batch_size 12 \
    --learning_rate 0.0002 \
    --num_workers 12 \
    --vlm_type mae \
    --mae_arch mae_base \
    --mae_finetune_type ln \
    --mae_ckpt_dir ./ckpt/ \
    --mae_load_ckpt True \
    --use_mem_gate True \
    --dropout 0.25 \
    --d_ff 1536 \
    --n_heads 12 \
    --percent 1 > logs/TimeVLM_mae_optimized_electricity_512_720_exp4.log 2>&1 &

echo "Running Optimized MAE-base experiment: pred_len=720, d_model=384, experiment_id=exp4"
echo "Log file: logs/TimeVLM_mae_optimized_electricity_512_720_exp4.log"
echo "Experiment started in background with PID: $!"

echo "All Optimized MAE-base experiments started in background!"
echo ""
echo "Monitor training progress:"
echo "- Use 'ps aux | grep python' to see running processes"
echo "- Use 'tail -f logs/*.log' to monitor specific experiments"
echo "- Checkpoints will be saved in './checkpoints/' directory"
echo ""
echo "Optimized Experiment configurations:"
echo "- Model: TimeVLM with MAE-base encoder"
echo "- Dataset: Electricity (321 variables)"
echo "- Sequence length: 512"
echo "- Prediction lengths: 96, 192, 336, 720"
echo "- Training epochs: 20-25 (increased from 15)"
echo "- Learning rates: 0.0002-0.0005 (decreased for better convergence)"
echo "- Batch sizes: 12-24 (optimized for memory)"
echo "- Model depth: 3-4 layers (increased capacity)"
echo "- Dropout: 0.2-0.25 (improved regularization)"
echo ""
echo "Key improvements over previous experiments:"
echo "✅ Higher model capacity (d_model: 256-384 vs 128-512)"
echo "✅ More training epochs (20-25 vs 15)"
echo "✅ Better learning rate scheduling (0.0002-0.0005 vs 0.001)"
echo "✅ Improved regularization (dropout: 0.2-0.25 vs 0.3)"
echo "✅ Memory-optimized batch sizes (12-24 vs 32)"
echo "✅ Increased feed-forward dimensions (d_ff: 1024-1536 vs 768)"
echo "✅ More attention heads (8-12 vs 8)"
echo "✅ Deeper encoder networks (3-4 layers vs 2)"