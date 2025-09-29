#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# MAE-base encoder configuration for electricity dataset - Experiment 4 retry with smaller batch size
model_name=TimeVLM
vlm_type=mae                    # Use MAE encoder instead of CLIP
mae_arch=mae_base               # Use MAE-base architecture
mae_finetune_type=ln            # Fine-tune only layer norm
gpu=0                           # Use GPU 0
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False              # Don't fine-tune VLM (MAE encoder)
batch_size=6                    # Further reduced to 6 to save more memory
num_workers=4                   # Further reduced to 4 to save more memory
learning_rate=0.0002            # Optimized learning rate
seq_len=512
label_len=64                    # Optimized label length
percent=1                       # 100% of training data
train_epochs=25                 # Optimized training epochs

# Create logs directory if it doesn't exist
if [ ! -d "logs" ]; then
    mkdir logs
fi

# Create checkpoints directory if it doesn't exist
if [ ! -d "checkpoints" ]; then
    mkdir -p checkpoints
fi

# Download MAE checkpoint if not exists
if [ ! -f "./ckpt/mae_visualize_vit_base.pth" ]; then
    echo "Downloading MAE-base checkpoint..."
    mkdir -p ./ckpt/
    wget -P ./ckpt/ https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth
    if [ $? -eq 0 ]; then
        echo "MAE-base checkpoint downloaded successfully"
    else
        echo "Failed to download MAE-base checkpoint. Please download manually:"
        echo "wget -P ./ckpt/ https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth"
        exit 1
    fi
else
    echo "MAE-base checkpoint already exists"
fi

# Experiment 4: Very long-term forecasting (720 steps, ~30 days for hourly data)
# Optimized configuration with reduced batch size to avoid memory issues
pred_len=720
d_model=384                     # Optimized model dimension
use_mem_gate=True               # Use memory gate
periodicity=24                  # Hourly data periodicity
dropout=0.25                    # Optimized dropout
experiment_id="exp4_v4"          # Version 4 with batch size 6

log_file="logs/${model_name}_mae_optimized_electricity_${seq_len}_${pred_len}_${experiment_id}.log"
echo "Running MAE-base experiment 4 with reduced batch size:"
echo "  pred_len=${pred_len}, d_model=${d_model}, batch_size=${batch_size} (further reduced)"
echo "Log file: ${log_file}"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity/electricity.csv \
  --model_id electricity_mae_optimized_${seq_len}_${pred_len}_${experiment_id} \
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
  --des 'MAE-base Experiment - Batch Size 8' \
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
  --percent $percent > $log_file 2>&1 &

echo "Experiment 4 started in background with PID: $!"
echo "Monitor progress with: tail -f ${log_file}"
echo ""
echo "Experiment configuration:"
echo "- Model: TimeVLM with MAE-base encoder"
echo "- Dataset: Electricity (321 variables)"
echo "- Sequence length: 512"
echo "- Prediction length: 720"
echo "- Model dimension: 384"
echo "- Batch size: 6 (further reduced from 12)"
echo "- Learning rate: 0.0002"
echo "- Training epochs: 25"
echo "- Workers: 4 (further reduced from 12)"