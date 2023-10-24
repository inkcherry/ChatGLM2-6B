PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
echo "!!!Before use this script, please change chatglm2-6b-local\config.json to torch_dtype to bfloat16."

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path chatglm2-6b-local\
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 20 \
    --max_target_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 100 \
    --save_steps 1000 \
    --learning_rate $LR \
    --use_habana \
    --use_lazy_mode \
    --preprocessing_num_workers 10 \
    --bf16 True \
    --use_lora \
    --lora_rank 64 \
    --ddp_find_unused_parameters=False \
    --gaudi_config_name gaudi_config.json \

    #  --pre_seq_len $PRE_SEQ_LEN \

