#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# Simple Zero-shot validation test
model_name=TimeVLM
vlm_type=mae
gpu=0
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=4
num_workers=2
learning_rate=0.0001
seq_len=512
label_len=96
percent=1
train_epochs=1

echo "Starting Simple Zero-shot Test..."
echo "================================="

# Test one combination: ETTh1 model -> ETTh2 data
source_data="ETTh1"
target_data="ETTh2"
pred_len="96"

echo "Testing: ${source_data} model -> ${target_data} data (pred_len=${pred_len})"

# Source model path
source_model_dir="checkpoints/long_term_forecast_mae_${source_data}_mae_reconstruction_512_${pred_len}_${source_data}_reconstruction_${pred_len}_TimeVLM_custom_ftM_sl512_ll96_pl${pred_len}_dm256_fs1.0_0"
source_model_path="${source_model_dir}/checkpoint.pth"

# Check if source model exists
if [ ! -f "$source_model_path" ]; then
    echo "ERROR: Source model not found: $source_model_path"
    exit 1
fi

# Test configuration
experiment_id="zero_shot_simple_${source_data}_to_${target_data}_${pred_len}"
log_file="logs/zero_shot_simple/${model_name}_mae_${experiment_id}.log"

echo "Source model: $source_model_path"
echo "Target data: $target_data"
echo "Log file: $log_file"

# Create log directory
mkdir -p logs/zero_shot_simple

# Copy the source model to a temporary location for testing
cp "$source_model_path" "/tmp/checkpoint_zero_shot.pth"

# Run zero-shot testing
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ${target_data}.csv \
  --model_id ${experiment_id} \
  --model $model_name \
  --data custom \
  --features M \
  --target OT \
  --freq h \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --d_model 256 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 512 \
  --factor 3 \
  --moving_avg 25 \
  --distil \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des "Zero-shot Simple: ${source_data} model on ${target_data} data" \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --train_epochs $train_epochs \
  --patience 1 \
  --lradj type1 \
  --image_size $image_size \
  --norm_const $norm_const \
  --periodicity 24 \
  --three_channel_image $three_channel_image \
  --finetune_vlm $finetune_vlm \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers \
  --vlm_type $vlm_type \
  --mae_arch mae_base \
  --mae_finetune_type ln \
  --mae_ckpt_dir "./ckpt/" \
  --mae_load_ckpt True \
  --mae_decoder_enabled False \
  --use_optimized_mae False \
  --use_adaptive_norm True \
  --use_global_features True \
  --feature_fusion True \
  --learnable_image True \
  --save_images False \
  --use_cross_attention True \
  --w_out_visual False \
  --w_out_text False \
  --w_out_query False \
  --visualize_embeddings False \
  --use_mem_gate True \
  --use_reconstruction_mae True \
  --use_dual_path_reconstruction False \
  --dropout 0.1 \
  --percent $percent \
  --llm_model GPT2 \
  --llm_dim 768 \
  --stride 8 \
  --padding 8 \
  --patch_len 16 \
  --llm_layers 1 \
  --prompt_domain 0 \
  --align_const 0.4 \
  --wo_ts 0 \
  --use_dtw False \
  --augmentation_ratio 0 \
  --seed 2024 > $log_file 2>&1

echo "Waiting for test completion..."
sleep 10

# Check results
if [ -f "$log_file" ]; then
    echo "Log file created. Checking for results..."
    mse_result=$(grep -E "MSE: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")
    mae_result=$(grep -E "MAE: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")

    if [ -n "$mse_result" ]; then
        echo "✓ SUCCESS: MSE=$mse_result, MAE=$mae_result"
        echo "${source_data}->${target_data}_${pred_len}: MSE=$mse_result, MAE=$mae_result" > logs/zero_shot_simple/result.txt
    else
        echo "✗ FAILED: No MSE/MAE results found"
        echo "Checking log for errors..."
        tail -20 "$log_file"
    fi
else
    echo "✗ FAILED: No log file created"
fi

echo ""
echo "Simple zero-shot test completed!"