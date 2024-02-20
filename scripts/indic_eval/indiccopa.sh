#!/bin/bash

source ./scripts/indic_eval/common_vars.sh


for model_name_or_path in "${model_names[@]}"; do
    model_name=${model_name_or_path##*/}
    TASK_NAME=indiccopa
    LANG=hi
    
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${LANG}"
    FILE=$FOLDER/metrics.json

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""
    fi

    check_file_existence=true
    template_format="eval.templates.create_prompt_with_chatml_format"
    if echo "$model_name" | grep -qi "Airavata"; then
        template_format="eval.templates.create_prompt_with_tulu_chat_format"
        check_file_existence=false
    fi
    if echo "$model_name" | grep -qi "OpenHathi-7B-Hi-v0.1-Base"; then
        template_format="eval.templates.create_prompt_with_llama2_chat_format"
        check_file_existence=false
    fi

    if [ "$check_file_existence" = false ] || [ ! -f "$FILE" ]; then

    if [ ! -f "$FILE" ]; then
        # zero-shot
        python3 -m eval.indiccopa.run_eval \
            --ntrain 0 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 8 \
            --use_chat_format \
            --chat_formatting_function $template_format \
            $awq_param

    fi

    LANG=5short
    FOLDER="${FOLDER_BASE}/${TASK_NAME}/${model_name}/${LANG}"
    FILE=$FOLDER/metrics.json

    if echo "$model_name" | grep -qi "awq"; then
        awq_param="--awq"
    else
        awq_param=""

    if [ ! -f "$FILE" ]; then
        # 5-shot
        python3 -m eval.indiccopa.run_eval \
            --ntrain 5 \
            --save_dir $FOLDER \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path $model_name_or_path \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_chatml_format \
            $awq_param
    fi
done