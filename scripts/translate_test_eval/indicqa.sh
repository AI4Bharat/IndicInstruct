# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=1

base_dir="/data/jaygala/llama_ckpts/llama-v2-hf/"

# Llama-2 7B Base Checkpoint
model_name_or_path="${base_dir}/llama-2-7b"
echo "evaluating llama 2 base on indicqa ..."

# # no-context
python3 -m eval.indicqa.run_eval_v2 \
    --ntrain 1 \
    --max_context_length 768 \
    --no_context \
    --save_dir "results/translate_test/llama-2/indicqa/llama-2-7b-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# with context
python3 -m eval.indicqa.run_eval_v2 \
    --ntrain 1 \
    --max_context_length 768 \
    --save_dir "results/translate_test/llama-2/indicqa/llama-2-7b-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 2


# Llama-2 7B Chat Checkpoint

model_name_or_path="${base_dir}/llama-2-7b-chat"
echo "evaluating llama 2 7b chat on indicqa ..."

# no-context
python3 -m eval.indicqa.run_eval \
    --ntrain 1 \
    --max_context_length 768 \
    --no_context \
    --save_dir "results/translate_test/llama-2/indicqa/llama-2-7b-chat-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 16 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# with context
python3 -m eval.indicqa.run_eval \
    --ntrain 1 \
    --max_context_length 768 \
    --save_dir "results/translate_test/llama-2/indicqa/llama-2-7b-chat-1shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 2 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
