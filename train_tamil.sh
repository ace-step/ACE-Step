# train_tamil.sh
#!/bin/bash

echo "🚀 Starting Tamil TTS LoRA Training"

python train_tamil_lora.py \
    --tamil_dataset_path "tamil_dataset_ace_step/hf_dataset" \
    --lora_config_path "config/tamil_lora_config.json" \
    --exp_name "tamil_tts_v1" \
    --learning_rate 1e-4 \
    --max_steps 50000 \
    --every_plot_step 1000 \
    --devices 1 \
    --num_workers 4 \
    --batch_size 2 \
    --checkpoint_dir "./checkpoints" \
    --logger_dir "./exps/logs"