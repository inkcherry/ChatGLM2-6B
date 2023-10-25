export CUDA_VISIBLE_DEVICES=0

PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
# use python main.py  or  ddp_find_unused_parameters for https://github.com/tloen/alpaca-lora/issues/301
# python main.py \
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path chatglm2-6b-local \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 20 \
    --max_target_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 500 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --use_lora \
    --lora_rank 64 \
    --lora_module 'dense' 'query_key_value' \
    --predict_with_generate \
    --ddp_find_unused_parameters=False \
    --fp16 True \
    # --quantization_bit 4
    # --pre_seq_len $PRE_SEQ_LEN \


