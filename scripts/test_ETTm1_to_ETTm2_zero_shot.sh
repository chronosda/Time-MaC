#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# ETTm1 -> ETTm2 Zero-shot validation for all prediction lengths
echo "Starting ETTm1 -> ETTm2 Zero-shot Validation..."
echo "==============================================="

source_data="ETTm1"
target_data="ETTm2"
pred_lengths=(96 192 336 720)

# Create log directory
mkdir -p logs/zero_shot_ETTm1_to_ETTm2

echo "Testing all prediction lengths: ${pred_lengths[*]}"

for pred_len in "${pred_lengths[@]}"; do
    echo ""
    echo "==============================================="
    echo "Testing: ETTm1 model -> ETTm2 data (pred_len=${pred_len})"
    echo "==============================================="

    # Source model path
    source_model_dir="checkpoints/long_term_forecast_mae_${source_data}_mae_reconstruction_512_${pred_len}_${source_data}_reconstruction_${pred_len}_TimeVLM_custom_ftM_sl512_ll96_pl${pred_len}_dm256_fs1.0_0"
    source_model_path="${source_model_dir}/checkpoint.pth"

    # Check if source model exists
    if [ ! -f "$source_model_path" ]; then
        echo "ERROR: Source model not found: $source_model_path"
        echo "ETTm1->ETTm2_${pred_len}: MODEL_NOT_FOUND" >> logs/zero_shot_ETTm1_to_ETTm2/results.txt
        continue
    fi

    # Test configuration
    experiment_id="zero_shot_ETTm1_to_ETTm2_${pred_len}"
    log_file="logs/zero_shot_ETTm1_to_ETTm2/${experiment_id}.log"

    echo "Source model: $source_model_path"
    echo "Target data: $target_data"
    echo "Prediction length: $pred_len"
    echo "Log file: $log_file"

    # Run zero-shot testing with DTW disabled for faster results
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
      --des "Zero-shot: ETTm1 model on ETTm2 data (pred_len=${pred_len})" \
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
        echo "Test completed. Analyzing results..."

        # Look for Test Loss during training
        test_loss=$(grep -E "Test Loss: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")

        # Look for final mse/mae results
        mse_result=$(grep -E "mse: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")
        mae_result=$(grep -E "mae: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")

        # Get test shape info
        test_shape=$(grep -E "test shape: \([0-9]+, [0-9]+, [0-9]+\)" "$log_file" | tail -1)

        if [ -n "$mse_result" ]; then
            echo "✓ COMPLETED: pred_len=${pred_len}, MSE=$mse_result, MAE=$mae_result"
            echo "Test shape: $test_shape"
            echo "ETTm1->ETTm2_${pred_len}: MSE=$mse_result, MAE=$mae_result, Test_Loss=$test_loss" >> logs/zero_shot_ETTm1_to_ETTm2/results.txt
        elif [ -n "$test_loss" ]; then
            echo "✓ PARTIAL: pred_len=${pred_len}, Test Loss=$test_loss (DTW calculation may be incomplete)"
            echo "Test shape: $test_shape"
            echo "ETTm1->ETTm2_${pred_len}: MSE=N/A, MAE=N/A, Test_Loss=$test_loss" >> logs/zero_shot_ETTm1_to_ETTm2/results.txt
        else
            echo "✗ FAILED: No results found in log"
            echo "Checking for errors..."
            tail -10 "$log_file"
            echo "ETTm1->ETTm2_${pred_len}: FAILED" >> logs/zero_shot_ETTm1_to_ETTm2/results.txt
        fi
    else
        echo "✗ FAILED: No log file created"
        echo "ETTm1->ETTm2_${pred_len}: NO_LOG" >> logs/zero_shot_ETTm1_to_ETTm2/results.txt
    fi
done

echo ""
echo "==============================================="
echo "ETTm1 -> ETTm2 Zero-shot Validation Summary"
echo "==============================================="

if [ -f "logs/zero_shot_ETTm1_to_ETTm2/results.txt" ]; then
    echo "Results Summary:"
    echo "================"
    cat logs/zero_shot_ETTm1_to_ETTm2/results.txt

    echo ""
    echo "Performance Comparison:"
    echo "======================="
    echo "ETTm1 Native Performance:"
    grep -E "ETTm1.*reconstruction.*96.*log$" logs/TimeVLM_mae_reconstruction_ETTm1_512_96_ETTm1_reconstruction_96.log | xargs grep -E "mse:|mae:" | tail -1
    grep -E "ETTm1.*reconstruction.*192.*log$" logs/TimeVLM_mae_reconstruction_ETTm1_512_192_ETTm1_reconstruction_192.log | xargs grep -E "mse:|mae:" | tail -1
    grep -E "ETTm1.*reconstruction.*336.*log$" logs/TimeVLM_mae_reconstruction_ETTm1_512_336_ETTm1_reconstruction_336.log | xargs grep -E "mse:|mae:" | tail -1
    grep -E "ETTm1.*reconstruction.*720.*log$" logs/TimeVLM_mae_reconstruction_ETTm1_512_720_ETTm1_reconstruction_720.log | xargs grep -E "mse:|mae:" | tail -1

    echo ""
    echo "ETTm2 Native Performance:"
    grep -E "ETTm2.*reconstruction.*96.*log$" logs/TimeVLM_mae_reconstruction_ETTm2_512_96_ETTm2_reconstruction_96.log | xargs grep -E "mse:|mae:" | tail -1
    grep -E "ETTm2.*reconstruction.*192.*log$" logs/TimeVLM_mae_reconstruction_ETTm2_512_192_ETTm2_reconstruction_192.log | xargs grep -E "mse:|mae:" | tail -1
    grep -E "ETTm2.*reconstruction.*336.*log$" logs/TimeVLM_mae_reconstruction_ETTm2_512_336_ETTm2_reconstruction_336.log | xargs grep -E "mse:|mae:" | tail -1
    grep -E "ETTm2.*reconstruction.*720.*log$" logs/TimeVLM_mae_reconstruction_ETTm2_512_720_ETTm2_reconstruction_720.log | xargs grep -E "mse:|mae:" | tail -1
else
    echo "No results file found"
fi

echo ""
echo "Zero-shot validation completed!"
echo "Results saved to logs/zero_shot_ETTm1_to_ETTm2/"