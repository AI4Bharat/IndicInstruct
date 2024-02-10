# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


model_names=(
    "manishiitg/open-aditi-hi-v2-awq"
    "manishiitg/open-aditi-hi-v1-awq"
    "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"
)
FOLDER_BASE=/sky-notebook/eval-results/indicqa


for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=indicqa
    NUM_SHOTS=no-context
    
    # FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    # FILE=$FOLDER/metrics.json

    # if [ ! -f "$FILE" ]; then
    #     # no-context
    #     python3 -m eval.indicqa.run_translate_test_eval \
    #         --ntrain 0 \
    #         --max_context_length 768 \
    #         --no_context \
    #         --save_dir $FOLDER \
    #         --model_name_or_path $model_name_or_path \
    #         --tokenizer_name_or_path $model_name_or_path \
    #         --eval_batch_size 4 \
    #         --use_chat_format \
    #         --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
    #         --awq \
    #         --use_vllm
    # else
    #     cat "$FILE"
    # fi

    NUM_SHOTS=with-context
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${NUM_SHOTS}"
    FILE=$FOLDER/metrics.json
    echo "evaluating $model_name base on $TASK_NAME $NUM_SHOTS ..."

    if [ ! -f "$FILE" ]; then
        # with context
        python3 -m eval.indicqa.run_translate_test_eval \
            --ntrain 0 \
            --max_context_length 3750 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 2 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            --awq \
            --use_vllm
    else
        cat "$FILE"
    fi
done
