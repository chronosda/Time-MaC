#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# Test reconstruction-oriented MAE architectures
model_name=TimeVLM
vlm_type=mae                          # Use MAE encoder
gpu=0
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=4                          # Smaller batch for testing
num_workers=2                         # Reduced workers
learning_rate=0.0001                  # Lower learning rate for testing
seq_len=96                            # Shorter sequence for quick testing
label_len=32
percent=1
train_epochs=5                        # Fewer epochs for testing

# Create directories
mkdir -p logs checkpoints

# Test different MAE architectures
test_types=("standard" "optimized" "reconstruction" "dual_path")
test_results=()

echo "Starting comprehensive MAE architecture comparison test..."
echo "=========================================================="

for test_type in "${test_types[@]}"; do
    echo ""
    echo "Testing ${test_type} MAE architecture..."

    # Set parameters based on test type
    case $test_type in
        "standard")
            use_optimized_mae=False
            use_reconstruction_mae=False
            use_dual_path_reconstruction=False
            mae_finetune_type="ln"
            ;;
        "optimized")
            use_optimized_mae=True
            use_reconstruction_mae=False
            use_dual_path_reconstruction=False
            mae_finetune_type="enhanced"
            ;;
        "reconstruction")
            use_optimized_mae=False
            use_reconstruction_mae=True
            use_dual_path_reconstruction=False
            mae_finetune_type="ln"
            ;;
        "dual_path")
            use_optimized_mae=False
            use_reconstruction_mae=False
            use_dual_path_reconstruction=True
            mae_finetune_type="ln"
            ;;
    esac

    # Test configuration
    pred_len=24                        # Short prediction for quick testing
    d_model=256                        # Smaller model for testing
    use_mem_gate=True
    periodicity=24
    dropout=0.1
    experiment_id="${test_type}_test"

    log_file="logs/${model_name}_mae_${test_type}_${seq_len}_${pred_len}_${experiment_id}.log"
    echo "Log file: ${log_file}"

    # Run the experiment
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity/electricity.csv \
      --model_id electricity_mae_${test_type}_${seq_len}_${pred_len}_${experiment_id} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --d_model $d_model \
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
      --des 'MAE Architecture Comparison Test' \
      --itr 1 \
      --gpu $gpu \
      --use_amp \
      --train_epochs $train_epochs \
      --patience 3 \
      --lradj type1 \
      --image_size $image_size \
      --norm_const $norm_const \
      --periodicity $periodicity \
      --three_channel_image True \
      --finetune_vlm $finetune_vlm \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --num_workers $num_workers \
      --vlm_type $vlm_type \
      --mae_arch mae_base \
      --mae_finetune_type $mae_finetune_type \
      --mae_ckpt_dir "./ckpt/" \
      --mae_load_ckpt True \
      --mae_decoder_enabled False \
      --learnable_image True \
      --save_images False \
      --use_cross_attention True \
      --w_out_visual False \
      --w_out_text False \
      --w_out_query False \
      --visualize_embeddings False \
      --use_mem_gate $use_mem_gate \
      --dropout $dropout \
      --percent $percent \
      --use_optimized_mae $use_optimized_mae \
      --use_reconstruction_mae $use_reconstruction_mae \
      --use_dual_path_reconstruction $use_dual_path_reconstruction > $log_file 2>&1 &

    test_pid=$!
    echo "Started ${test_type} test with PID: $test_pid"

    # Wait for completion and collect results
    wait $test_pid

    # Extract results from log
    if [ -f "$log_file" ]; then
        mse_result=$(grep -E "MSE: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")
        if [ -n "$mse_result" ]; then
            test_results+=("${test_type}: ${mse_result}")
            echo "${test_type} MAE test completed. MSE: ${mse_result}"
        else
            test_results+=("${test_type}: FAILED")
            echo "${test_type} MAE test failed. Check log: ${log_file}"
        fi
    else
        test_results+=("${test_type}: NO_LOG")
        echo "${test_type} MAE test failed. No log file found."
    fi
done

echo ""
echo "=========================================================="
echo "MAE Architecture Comparison Results:"
echo "=========================================================="

for result in "${test_results[@]}"; do
    echo "  $result"
done

echo ""
echo "Detailed logs available in logs/ directory"
echo "Test completed!"