#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
repo_root="$(dirname "$script_dir")"
cd "$repo_root"
export TOKENIZERS_PARALLELISM=false

# Strict zero-shot: train on ETTh1 only; evaluate on ETTh2 (no target-domain training)

pred_lengths=(96 192 336 720)
log_dir="logs/strict_zero_shot_ETTh1_to_ETTh2"
mkdir -p "$log_dir"
results_file="$log_dir/results.txt"
echo "# Strict Zero-shot: ETTh1 -> ETTh2" > "$results_file"
date >> "$results_file"

for pred_len in "${pred_lengths[@]}"; do
  exp_id="ZS_ETTh1_to_ETTh2_${pred_len}"
  log_file="$log_dir/${exp_id}.log"

  echo "Running ${exp_id} ..."

  python -u run.py \
    --task_name zero_shot_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data ETTh1 \
    --data_path ETTh1.csv \
    --target_data ETTh2 \
    --target_root_path ./dataset/ \
    --target_data_path ETTh2.csv \
    --model_id ${exp_id} \
    --model TimeVLM \
    --features M \
    --freq h \
    --seq_len 512 \
    --label_len 96 \
    --pred_len ${pred_len} \
    --d_model 256 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 512 \
    --factor 3 \
    --moving_avg 25 \
    --distil \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des "Strict Zero-shot: ETTh1->ETTh2 (pred_len=${pred_len})" \
    --itr 1 \
    --gpu 0 \
    --use_amp \
    --train_epochs 10 \
    --patience 5 \
    --image_size 56 \
    --norm_const 0.4 \
    --periodicity 24 \
    --three_channel_image True \
    --finetune_vlm False \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_workers 4 \
    --vlm_type mae \
    --use_mem_gate True \
    --use_reconstruction_mae True \
    --use_dual_path_reconstruction False \
    --percent 1 > "$log_file" 2>&1

  mse=$(grep -E "MSE: [0-9]+\.[0-9]+" "$log_file" | tail -1 | sed -E 's/.*MSE: ([0-9]+\.[0-9]+).*/\1/') || true
  mae=$(grep -E "MAE: [0-9]+\.[0-9]+" "$log_file" | tail -1 | sed -E 's/.*MAE: ([0-9]+\.[0-9]+).*/\1/') || true
  echo "${exp_id}: MSE=${mse:-N/A}, MAE=${mae:-N/A}" | tee -a "$results_file"
done

echo "Done. Summary at: $results_file"
