#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# ETTh1 -> ETTh2 Zero-shot validation
echo "Starting ETTh1 -> ETTh2 Zero-shot Validation..."
echo "==============================================="

source_data="ETTh1"
target_data="ETTh2"
pred_lengths=(96 192 336 720)

for pred_len in "${pred_lengths[@]}"; do
    echo ""
    echo "Testing: ETTh1 model -> ETTh2 data (pred_len=${pred_len})"

    # Source model path
    source_model_dir="checkpoints/long_term_forecast_mae_${source_data}_mae_reconstruction_512_${pred_len}_${source_data}_reconstruction_${pred_len}_TimeVLM_custom_ftM_sl512_ll96_pl${pred_len}_dm256_fs1.0_0"
    source_model_path="${source_model_dir}/checkpoint.pth"

    # Check if source model exists
    if [ ! -f "$source_model_path" ]; then
        echo "ERROR: Source model not found: $source_model_path"
        continue
    fi

    # Test configuration
    experiment_id="zero_shot_ETTh1_to_ETTh2_${pred_len}"
    log_file="logs/zero_shot_ETTh1_to_ETTh2/${experiment_id}.log"

    echo "Source model: $source_model_path"
    echo "Target data: $target_data"
    echo "Log file: $log_file"

    # Create log directory
    mkdir -p logs/zero_shot_ETTh1_to_ETTh2

    # Run zero-shot testing
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ${target_data}.csv \
      --model_id ${experiment_id} \
      --model TimeVLM \
      --data custom \
      --features M \
      --target OT \
      --freq h \
      --seq_len 512 \
      --label_len 96 \
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
      --des "Zero-shot: ETTh1 model on ETTh2 data (pred_len=${pred_len})" \
      --itr 1 \
      --gpu 0 \
      --use_amp \
      --train_epochs 1 \
      --patience 1 \
      --lradj type1 \
      --image_size 56 \
      --norm_const 0.4 \
      --periodicity 24 \
      --three_channel_image True \
      --finetune_vlm False \
      --batch_size 4 \
      --learning_rate 0.0001 \
      --num_workers 2 \
      --vlm_type mae \
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
      --percent 1 \
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
      --seed 2024 > $log_file 2>&1 &

    test_pid=$!
    echo "Started test with PID: $test_pid"

    # Wait for completion
    wait $test_pid

    # Check results
    if [ -f "$log_file" ]; then
        mse_result=$(grep -E "MSE: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")
        mae_result=$(grep -E "MAE: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")

        if [ -n "$mse_result" ]; then
            echo "✓ COMPLETED: pred_len=${pred_len}, MSE=$mse_result, MAE=$mae_result"
            echo "ETTh1->ETTh2_${pred_len}: MSE=$mse_result, MAE=$mae_result" >> logs/zero_shot_ETTh1_to_ETTh2/results.txt
        else
            echo "✗ FAILED: No results found in log"
            echo "Checking for errors..."
            tail -10 "$log_file"
            echo "ETTh1->ETTh2_${pred_len}: FAILED" >> logs/zero_shot_ETTh1_to_ETTh2/results.txt
        fi
    else
        echo "✗ FAILED: No log file created"
        echo "ETTh1->ETTh2_${pred_len}: NO_LOG" >> logs/zero_shot_ETTh1_to_ETTh2/results.txt
    fi
done

echo ""
echo "==============================================="
echo "ETTh1 -> ETTh2 Zero-shot Validation Summary"
echo "==============================================="

if [ -f "logs/zero_shot_ETTh1_to_ETTh2/results.txt" ]; then
    echo "Results:"
    cat logs/zero_shot_ETTh1_to_ETTh2/results.txt
else
    echo "No results file found"
fi

echo ""
echo "Zero-shot validation completed!"