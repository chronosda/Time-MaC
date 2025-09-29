#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# 重建导向MAE架构训练electricity数据集
model_name=TimeVLM
vlm_type=mae                          # 使用MAE编码器
gpu=0
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=6                          # 适中的batch size
num_workers=4
learning_rate=0.0001
seq_len=512                           # 长序列预测
label_len=96
percent=1
train_epochs=10                       # 充分训练

# 创建目录
mkdir -p logs checkpoints

echo "Training Reconstruction-Oriented MAE on Electricity Dataset"
echo "============================================================"

# 多个预测长度进行实验
pred_lengths=(96 192 336 720)

for pred_len in "${pred_lengths[@]}"; do
    echo ""
    echo "Training for prediction length: ${pred_len}"

    # 重建导向MAE配置
    use_optimized_mae=False
    use_reconstruction_mae=True        # 启用重建导向MAE
    use_dual_path_reconstruction=False
    mae_finetune_type="ln"

    # 模型配置
    d_model=256                        # 中等模型大小
    use_mem_gate=True
    periodicity=24                      # 电力数据的日周期性
    dropout=0.1
    experiment_id="electricity_reconstruction_${pred_len}"

    log_file="logs/${model_name}_mae_reconstruction_electricity_${seq_len}_${pred_len}_${experiment_id}.log"
    echo "Log file: ${log_file}"

    # 运行实验
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity/electricity.csv \
      --model_id electricity_mae_reconstruction_${seq_len}_${pred_len}_${experiment_id} \
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
      --des 'Reconstruction-Oriented MAE on Electricity Dataset' \
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
      --use_dual_path_reconstruction $use_dual_path_reconstruction > $log_file 2>&1 &

    train_pid=$!
    echo "Started electricity reconstruction MAE training (pred_len=$pred_len) with PID: $train_pid"

    # 等待当前训练完成
    wait $train_pid

    # 提取结果
    if [ -f "$log_file" ]; then
        test_loss=$(grep -E "Test Loss: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")
        if [ -n "$test_loss" ]; then
            echo "Electricity reconstruction MAE (pred_len=$pred_len) completed. Test Loss: $test_loss"
        else
            echo "Electricity reconstruction MAE (pred_len=$pred_len) failed. Check log: $log_file"
        fi
    else
        echo "Electricity reconstruction MAE (pred_len=$pred_len) failed. No log file found."
    fi
done

echo ""
echo "============================================================"
echo "Electricity dataset training completed!"
echo "Results summary:"
for pred_len in "${pred_lengths[@]}"; do
    log_file="logs/${model_name}_mae_reconstruction_electricity_${seq_len}_${pred_len}_electricity_reconstruction_${pred_len}.log"
    if [ -f "$log_file" ]; then
        test_loss=$(grep -E "Test Loss: [0-9]+\.[0-9]+" "$log_file" | tail -1 | grep -o "[0-9]+\.[0-9]+")
        echo "  Prediction length $pred_len: Test Loss = ${test_loss:-'N/A'}"
    fi
done