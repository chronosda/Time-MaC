#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# MAE-base encoder configuration for electricity dataset
model_name=TimeVLM
vlm_type=mae                    # Use MAE encoder instead of CLIP
mae_arch=mae_base               # Use MAE-base architecture
mae_finetune_type=ln            # Fine-tune only layer norm
gpu=0                           # Use GPU 0
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False              # Don't fine-tune VLM (MAE encoder)
batch_size=32
num_workers=32
learning_rate=0.001
seq_len=512
percent=1                       # 100% of training data
train_epochs=15

# Create logs directory if it doesn't exist
if [ ! -d "logs" ]; then
    mkdir logs
fi

# Create checkpoints directory if it doesn't exist
if [ ! -d "checkpoints" ]; then
    mkdir -p checkpoints
fi

# Function to run MAE-base electricity experiment
run_mae_electricity_experiment() {
    local pred_len=$1
    local d_model=$2
    local use_mem_gate=$3
    local periodicity=$4
    local dropout=$5
    local experiment_id=$6

    # Determine task name based on percent
    local task_name="few_shot_forecast"
    if [ "$percent" = "1" ]; then
        task_name="long_term_forecast"
    fi

    log_file="logs/${model_name}_mae_electricity_${seq_len}_${pred_len}_${experiment_id}.log"
    echo "Running MAE-base experiment: pred_len=${pred_len}, d_model=${d_model}, experiment_id=${experiment_id}"
    echo "Log file: ${log_file}"

    python -u run.py \
      --task_name $task_name \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity/electricity.csv \
      --model_id electricity_mae_${seq_len}_${pred_len}_${experiment_id} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --d_model $d_model \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'MAE-base Experiment' \
      --itr 1 \
      --gpu $gpu \
      --use_amp \
      --train_epochs $train_epochs \
      --image_size $image_size \
      --norm_const $norm_const \
      --periodicity $periodicity \
      --three_channel_image $three_channel_image \
      --finetune_vlm $finetune_vlm \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --num_workers $num_workers \
      --vlm_type $vlm_type \
      --mae_arch $mae_arch \
      --mae_finetune_type $mae_finetune_type \
      --mae_ckpt_dir "./ckpt/" \
      --mae_load_ckpt True \
      --use_mem_gate $use_mem_gate \
      --dropout $dropout \
      --percent $percent > $log_file 2>&1 &

    echo "Experiment started in background with PID: $!"
    echo "Monitor progress with: tail -f ${log_file}"
}

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

# Run multiple experiments with different configurations
echo "Starting MAE-base training experiments for electricity dataset..."

# Experiment 1: Short-term forecasting (96 steps, ~4 hours for hourly data)
run_mae_electricity_experiment 96 128 True 24 0.3 "exp1"

# Experiment 2: Medium-term forecasting (192 steps, ~8 hours for hourly data)
run_mae_electricity_experiment 192 128 True 24 0.3 "exp2"

# Experiment 3: Long-term forecasting (336 steps, ~14 days for hourly data)
run_mae_electricity_experiment 336 256 True 24 0.3 "exp3"

# Experiment 4: Very long-term forecasting (720 steps, ~30 days for hourly data)
run_mae_electricity_experiment 720 512 True 24 0.3 "exp4"

echo "All MAE-base experiments started in background!"
echo ""
echo "Monitor training progress:"
echo "- Use 'ps aux | grep python' to see running processes"
echo "- Use 'tail -f logs/*.log' to monitor specific experiments"
echo "- Checkpoints will be saved in './checkpoints/' directory"
echo ""
echo "Experiment configurations:"
echo "- Model: TimeVLM with MAE-base encoder"
echo "- Dataset: Electricity (321 variables)"
echo "- Sequence length: 512"
echo "- Prediction lengths: 96, 192, 336, 720"
echo "- Training epochs: 15"
echo "- Batch size: 32"
echo "- Learning rate: 0.001"