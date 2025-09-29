#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# 单个重建导向MAE训练脚本
model_name=TimeVLM
vlm_type=mae
gpu=0
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=4
num_workers=4
learning_rate=0.0001
seq_len=512
label_len=96
percent=1
train_epochs=10

# 创建目录
mkdir -p logs checkpoints

# 从命令行参数获取预测长度
if [ -z "$1" ]; then
    echo "Usage: $0 <pred_len> [experiment_suffix]"
    exit 1
fi

pred_len=$1
experiment_suffix=${2:-"reconstruction"}

echo "Training Reconstruction-Oriented MAE on Electricity Dataset"
echo "============================================================"
echo "Prediction length: ${pred_len}"
echo "Experiment suffix: ${experiment_suffix}"

# 重建导向MAE配置
use_optimized_mae=False
use_reconstruction_mae=True
use_dual_path_reconstruction=False
mae_finetune_type="ln"

# 模型配置
d_model=256
use_mem_gate=True
periodicity=24
dropout=0.1
experiment_id="ETTh2_${experiment_suffix}_${pred_len}"

log_file="logs/${model_name}_mae_reconstruction_ETTh2_${seq_len}_${pred_len}_${experiment_id}.log"
echo "Log file: ${log_file}"

# 运行实验
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_mae_${experiment_suffix}_${seq_len}_${pred_len}_${experiment_id} \
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
  --des "Reconstruction-Oriented MAE on Electricity Dataset (pred_len=${pred_len})" \
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

echo "Training completed. Log file: ${log_file}"

# 提取结果
if [ -f "$log_file" ]; then
    test_loss=$(grep -E "Test Loss: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")
    if [ -n "$test_loss" ]; then
        echo "Electricity reconstruction MAE (pred_len=$pred_len) completed. Test Loss: $test_loss"
    else
        echo "Training may have failed. Check log: $log_file"
    fi
else
    echo "Training failed. No log file found."
fi