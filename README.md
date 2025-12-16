CUDA_VISIBLE_DEVICES=0 swift infer   --model Qwen/Qwen2.5-7B-Instruct   --torch_dtype bfloat16   --max_length 2048   --system 'You are a helpful assistant.'   --stream true

Run the original model


Fintuning train
 CUDA_VISIBLE_DEVICES=0 swift sft     --model Qwen/Qwen2.5-7B-Instruct     --train_type lora     --dataset /root/autodl-tmp/alpaca_dataset_top700.json     --torch_dtype float16     --num_train_epochs 4     --per_device_train_batch_size 1     --per_device_eval_batch_size 1     --learning_rate 5e-5     --lora_rank 8     --lora_alpha 32     --target_modules all-linear     --gradient_accumulation_steps 8    --eval_steps 50     --save_steps 50     --save_total_limit 2     --logging_steps 5     --max_length 12000     --output_dir /root/autodl-tmp/output     --system 'You are a helpful assistant.'     --warmup_ratio 0.05     --dataloader_num_workers 4     --model_author swift     --model_name swift-robot --resume_from_checkpoint /root/autodl-tmp/output/v30-20250825-223717/checkpoint-1150

Fintuning infer
infer
CUDA_VISIBLE_DEVICES=0 swift infer --model Qwen/Qwen2.5-7B-Instruct --adapters /root/autodl-tmp/output/v26-20250806-122310/checkpoint-348  --stream true

pip install transformers==4.44.2 ms-swift  sft版本
pip install ms-swift==3.5.0 transformers==4.46.0 trl==0.17.0 peft==0.11.0 torch deepspeed latex2sympy2 math_verify e2b-code-interpreter aiohttp python-dotenv grpo版本


MODELSCOPE_CACHE=/root/autodl-tmp/modelscope \
TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache \



run without the reward mechanism
python -m debugpy --listen 5678 /root/miniconda3/lib/python3.12/site-packages/swift/cli/rlhf.py \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset AI-MO/NuminaMath-TIR#5000 \
  --external_plugins /root/examples/train/grpo/plugin/plugin.py \
  --reward_funcs external_math_format external_math_acc \
  --reward_weights 0.6 0.4 \
  --vllm_gpu_memory_utilization 0.9 \
  --sleep_level 1 \
  --offload_model false \
  --offload_optimizer false \
  --log_completions true \
  --logging_steps 1 \
  --dataloader_num_workers 2 \
  --log_level debug \
  --per_device_eval_batch_size 8





CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --external_plugins /root/examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_math_format external_math_acc \
    --reward_weights 0.25 0.25 0.25 0.25 \
    --reward_model Qwen/Qwen2.5-3B-Instruct Shanghai_AI_Laboratory/internlm2-7b-reward \
    --reward_model_plugin genrm my_rmplugin \
    --reward_model_type qwen2_5 internlm2_reward \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --sleep_level 1 \
    --logging_steps 1 \
    --log_completions true \
    --log_level debug \
    --max_length 512 \
    --disable_tqdm false



CUDA_VISIBLE_DEVICES=0 swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset /root/autodl-tmp/rlhf_train_top300-3+1.parquet \
  --external_plugins /root/examples/train/grpo/plugin/plugin.py \
  --reward_model Qwen/Qwen2.5-3B-Instruct Shanghai_AI_Laboratory/internlm2-7b-reward \
  --reward_model_plugin genrm my_rmplugin \
  --reward_model_type qwen2_5 internlm2_reward \
  --reward_weights 0.5 0.5 \
  --per_device_eval_batch_size 8 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --sleep_level 1 \
  --logging_steps 1 \
  --log_completions true \
  --log_level debug \
  --max_length 512 \
  --disable_tqdm false \
  --output_dir /root/autodl-tmp/rlhf_output \
  --save_steps 10 \
  --save_total_limit 2\
  --resume_from_checkpoint /root/autodl-tmp/rlhf_output/v4-20250826-104751/checkpoint-720\
  --resume_only_model true


mkdir -p /root/autodl-tmp/logs; LOGFILE="/root/autodl-tmp/logs/rlhf_$(date +%F_%H-%M-%S).log"; nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 stdbuf -oL -eL swift rlhf --rlhf_type grpo --model Qwen/Qwen2.5-7B-Instruct --dataset /root/autodl-tmp/rlhf_train_top1000-3+1.parquet --external_plugins /root/examples/train/grpo/plugin/plugin.py --reward_model Qwen/Qwen2.5-3B-Instruct Shanghai_AI_Laboratory/internlm2-7b-reward --reward_model_plugin genrm my_rmplugin --reward_model_type qwen2_5 internlm2_reward --reward_weights 0.5 0.5 --per_device_eval_batch_size 4 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --sleep_level 1 --logging_steps 1 --log_completions true --log_level debug --max_length 512 --disable_tqdm false --learning_rate 5e-5 --num_generations 4 --output_dir /root/autodl-tmp/rlhf_output --save_steps 10 --save_total_limit 2 --save_safetensors true >>"$LOGFILE" 2>&1 < /dev/null & PID=$!; echo "$PID" > "${LOGFILE}.pid"; echo "PID=$PID  Log file=$LOGFILE"



