#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# Quick test of reconstruction-oriented MAE architectures
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
seq_len=48                            # Shorter sequence for quick testing
label_len=16
percent=1
train_epochs=3                        # Fewer epochs for quick testing

# Create directories
mkdir -p logs checkpoints

echo "Quick test of reconstruction-oriented MAE architectures..."
echo "=========================================================="

# Test only reconstruction and dual_path architectures
test_types=("reconstruction" "dual_path")

for test_type in "${test_types[@]}"; do
    echo ""
    echo "Testing ${test_type} MAE architecture..."

    # Set parameters based on test type
    case $test_type in
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
    pred_len=12                        # Short prediction for quick testing
    d_model=128                        # Smaller model for testing
    use_mem_gate=True
    periodicity=24
    dropout=0.1
    experiment_id="${test_type}_quick_test"

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
      --n_heads 4 \
      --e_layers 2 \
      --d_layers 1 \
      --d_ff 256 \
      --factor 3 \
      --moving_avg 25 \
      --distil \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Quick MAE Reconstruction Test' \
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
      --use_dual_path_reconstruction $use_dual_path_reconstruction > $log_file 2>&1

    test_pid=$!
    echo "Started ${test_type} test with PID: $test_pid"

    # Wait for completion
    wait $test_pid

    # Extract results from log
    if [ -f "$log_file" ]; then
        test_loss=$(grep -E "Test Loss: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")
        if [ -n "$test_loss" ]; then
            echo "${test_type} MAE test completed. Test Loss: ${test_loss}"
        else
            echo "${test_type} MAE test failed. Check log: ${log_file}"
        fi
    else
        echo "${test_type} MAE test failed. No log file found."
    fi
done

echo ""
echo "=========================================================="
echo "Quick test completed!"
echo "Check logs for detailed results"