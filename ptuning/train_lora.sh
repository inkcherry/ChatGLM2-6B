PRE_SEQ_LEN=128
LR=2e-2
# NUM_GPUS=1
echo "!!!Before use this script, please change chatglm2-6b-local\config.json to torch_dtype to bfloat16."
# python main.py \
# python gaudi_spawn.py --world_size 1  main.py \

# torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
python gaudi_spawn.py --world_size 2 --use_deepspeed  main.py\
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
    --max_target_length 8192 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 3000 \
    --logging_steps 3 \
    --save_steps 1000 \
    --learning_rate $LR \
    --use_habana \
    --use_lazy_mode \
    --preprocessing_num_workers 10 \
    --bf16 True \
    --use_hpu_graphs=False \
    --use_hpu_graphs_for_inference=False \
    --use_hpu_graphs_for_training=False \
    --gaudi_config_name gaudi_config.json \
    --use_lora \
    --lora_rank 64 \
    --lora_module 'query_key_value' \
    --gradient_checkpoint True\
        # --predict_with_generate \

    #--lora_module 'dense' 'query_key_value' \
    # --use_hpu_graphs 
    #  --pre_seq_len $PRE_SEQ_LEN \
    # --ddp_find_unused_parameters=False \



