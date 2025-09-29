#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# Simple comparison test: standard MAE vs reconstruction MAE
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
seq_len=48
label_len=16
percent=1
train_epochs=3

# Create directories
mkdir -p logs checkpoints

echo "Simple MAE Architecture Comparison Test"
echo "======================================="

# Test configurations
test_types=("standard" "reconstruction")

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
        "reconstruction")
            use_optimized_mae=False
            use_reconstruction_mae=True
            use_dual_path_reconstruction=False
            mae_finetune_type="ln"
            ;;
    esac

    # Test configuration
    pred_len=12
    d_model=128
    use_mem_gate=True
    periodicity=24
    dropout=0.1
    experiment_id="${test_type}_simple_test"

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
      --des 'Simple MAE Comparison Test' \
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

    echo "Started ${test_type} test..."
done

echo ""
echo "======================================="
echo "Test started in background!"
echo "Check logs for progress:"
echo "  - Standard MAE: logs/TimeVLM_mae_standard_48_12_standard_simple_test.log"
echo "  - Reconstruction MAE: logs/TimeVLM_mae_reconstruction_48_12_reconstruction_simple_test.log"
echo ""
echo "To check progress, use:"
echo "  tail -f logs/TimeVLM_mae_*.log"