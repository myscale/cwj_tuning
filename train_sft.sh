CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path YOUR_BASE_MODEL_PATH \
    --do_train \
    --dataset YOUR_TRAIN_DATASET_NAME \
    --template default \
    --finetuning_type lora \
    --lora_target all \
    --output_dir YOUR_OUTPUT_DIR \
    --overwrite_cache \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 200 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 \
    --quantization_bit 4
