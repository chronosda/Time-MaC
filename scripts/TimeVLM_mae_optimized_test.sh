#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# Optimized MAE encoder test configuration
model_name=TimeVLM
vlm_type=mae                          # Use MAE encoder
use_optimized_mae=True                # Use the optimized version
mae_arch=mae_base                     # MAE-base architecture
mae_finetune_type=enhanced            # Enhanced finetuning strategy
gpu=0
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=6                          # Memory optimized
num_workers=4                         # Reduced workers
learning_rate=0.0003                   # Slightly higher for enhanced training
seq_len=512
label_len=64
percent=1
train_epochs=20

# Enhanced parameters for optimized MAE
use_adaptive_norm=True                # Use adaptive normalization
use_global_features=True              # Use global features + CLS
feature_fusion=True                   # Enable feature fusion

# Create directories
mkdir -p logs checkpoints

# Download MAE checkpoint if needed
if [ ! -f "./ckpt/mae_visualize_vit_base.pth" ]; then
    echo "Downloading MAE-base checkpoint..."
    mkdir -p ./ckpt/
    wget -P ./ckpt/ https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth
fi

# Test with 96-step prediction (fastest to verify improvement)
pred_len=96
d_model=384
use_mem_gate=True
periodicity=24
dropout=0.25
experiment_id="optimized_test"

log_file="logs/${model_name}_mae_enhanced_${seq_len}_${pred_len}_${experiment_id}.log"
echo "Testing optimized MAE encoder..."
echo "Log file: ${log_file}"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity/electricity.csv \
  --model_id electricity_mae_enhanced_${seq_len}_${pred_len}_${experiment_id} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --d_model $d_model \
  --n_heads 12 \
  --e_layers 4 \
  --d_layers 2 \
  --d_ff 1536 \
  --factor 5 \
  --moving_avg 25 \
  --distil \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Optimized MAE Encoder Test' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --train_epochs $train_epochs \
  --patience 5 \
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
  --mae_arch $mae_arch \
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
  --use_adaptive_norm $use_adaptive_norm \
  --use_global_features $use_global_features \
  --feature_fusion $feature_fusion > $log_file 2>&1 &

echo "Optimized MAE test started with PID: $!"
echo "Monitor progress: tail -f ${log_file}"
echo ""
echo "Optimized MAE configuration:"
echo "- Enhanced finetuning: $mae_finetune_type"
echo "- Adaptive normalization: $use_adaptive_norm"
echo "- Global features: $use_global_features"
echo "- Feature fusion: $feature_fusion"
echo "- Batch size: $batch_size"