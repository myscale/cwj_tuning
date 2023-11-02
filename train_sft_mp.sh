CUDA_VISIBLE_DEVICES=2,3,6 torchrun --nproc_per_node 3 --standalone src/train_bash.py \
    --stage sft \
    --model_name_or_path /data/ccq/models/Baichuan2-13B-Chat \
    --do_train \
    --dataset dialog_inpainting_zh \
    --template default \
    --finetuning_type lora \
    --lora_target W_pack \
    --output_dir output/dialog_inpainting_Baichuan2_chatGPT_2 \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 200 \
    --learning_rate 5e-5 \
    --num_train_epochs 20.0 \
    --plot_loss \
    --fp16 \
    --quantization_bit 4