# --------------------------------------------------------------------------------
#				Main Configuration
#---------------------------------------------------------------------------------  
CUDA_VISIBLE_DEVICES=0

model_name_or_path=$1 # Model Path or Hugging Face name.
save_dir=$2 # Root directory to save results. Each task will be a subfolder where the respective results are stored. Caution: Ensure that the directory doesn't exist to avoid overwriting results.
chat_formatting_function=$3 # Optional. Options = [tulu, llama2, gemma]. If not provided, model will not use chat format.
eval_batch_size=10 # Batch Size to be used for evaluating.
run_eval=run_eval # Options = [run_eval, run_multi_turn_eval] Used for chat variants; Ignore for base model evaluations.
ntrains=(0) # Number of few-shot examples to be used for evaluation.
langs=(hi) # Languages to be evaluated.

if [ -z "$save_dir" ]; then
    save_dir="results/" # Variable is Empty, Take default folder as results/
fi


if [ -z "$chat_formatting_function" ]; then
    : # Do Nothing. Variable is Empty, No chat format will be used
else
    chat_formatting_function="--use_chat_format --chat_formatting_function eval.templates.create_prompt_with_${chat_formatting_function}_chat_format"
fi







# --------------------------------------------------------------------------------
#				Indic NLU
#---------------------------------------------------------------------------------    

# IndicSentiment
for ntrain in ${ntrains[@]}; do
for lang in ${langs[@]}; do

echo "Evaluating ${ntrain}-shot IndicSentiment (${lang})..."
echo "Results will be stored at $save_dir/indicsentiment/$lang/$ntrain-shot/"
python3 -m eval.indicsentiment.$run_eval \
	--ntrain $ntrain \
	--save_dir "${save_dir}/indicsentiment/${ntrain}-shot/" \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	--lang $lang \
	$chat_formatting_function
done
done

# IndicCOPA
for ntrain in ${ntrains[@]}; do
for lang in ${langs[@]}; do

echo "Evaluating ${ntrain}-shot IndicCOPA (${lang})..."
echo "Results will be stored at $save_dir/indiccopa/$lang/$ntrain-shot/"
python3 -m eval.indiccopa.$run_eval \
	--ntrain $ntrain \
	--save_dir "${save_dir}/indiccopa/${ntrain}-shot/" \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	--lang $lang \
	$chat_formatting_function
done
done

# IndicXNLI
for ntrain in ${ntrains[@]}; do
for lang in ${langs[@]}; do

echo "Evaluating ${ntrain}-shot IndicXNLI (${lang})..."
echo "Results will be stored at $save_dir/indicxnli/$lang/$ntrain-shot/"
python3 -m eval.indicxnli.$run_eval \
	--ntrain $ntrain \
	--save_dir "${save_dir}/indicxnli/${ntrain}-shot/" \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	--lang $lang \
	$chat_formatting_function
done
done

# IndicXParaphrase
for ntrain in ${ntrains[@]}; do
for lang in ${langs[@]}; do

echo "Evaluating ${ntrain}-shot IndicXParaphrase (${lang})..."
echo "Results will be stored at $save_dir/indicxparaphrase/$lang/$ntrain-shot/"
python3 -m eval.indicxparaphrase.$run_eval \
	--ntrain $ntrain \
	--save_dir "${save_dir}/indicxparaphrase/${ntrain}-shot/" \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	--lang $lang \
	$chat_formatting_function
done
done





# --------------------------------------------------------------------------------
#				English NLU
#---------------------------------------------------------------------------------

# MMLU
for ntrain in ${ntrains[@]}; do

echo "Evaluating ${ntrain}-shot MMLU..."
echo "Results will be stored at $save_dir/mmlu/$lang/$ntrain-shot/"
python3 -m eval.mmlu.$run_eval \
	--ntrain $ntrain \
    	--data_dir data/eval/mmlu \
	--save_dir "${save_dir}/mmlu/${ntrain}-shot/" \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function
done

# BoolQ
for ntrain in ${ntrains[@]}; do

echo "Evaluating ${ntrain}-shot BoolQ..."
echo "Results will be stored at $save_dir/boolq/$lang/$ntrain-shot/"
python3 -m eval.boolq.$run_eval \
	--ntrain $ntrain \
	--save_dir "${save_dir}/boolq/${ntrain}-shot/" \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function
done

# ARC-Easy
for ntrain in ${ntrains[@]}; do

echo "Evaluating ${ntrain}-shot ARC-Easy..."
echo "Results will be stored at $save_dir/arc-easy/$lang/$ntrain-shot/"
python3 -m eval.arc.$run_eval \
	--ntrain $ntrain \
    	--dataset "ai2_arc" \
    	--subset "easy" \
	--save_dir "${save_dir}/arc-easy/${ntrain}-shot/" \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function
done



# ARC-Challenge
for ntrain in ${ntrains[@]}; do

echo "Evaluating ${ntrain}-shot ARC-Challenge..."
echo "Results will be stored at $save_dir/arc-challenge/$lang/$ntrain-shot/"
python3 -m eval.arc.$run_eval \
	--ntrain $ntrain \
    	--dataset "ai2_arc" \
    	--subset "challenge" \
	--save_dir "${save_dir}/arc-challenge/${ntrain}-shot/" \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function
done

# Hellaswag
for ntrain in ${ntrains[@]}; do

echo "Evaluating ${ntrain}-shot Hellaswag..."
echo "Results will be stored at $save_dir/hellaswag/$lang/$ntrain-shot/"
python3 -m eval.hellaswag.$run_eval \
	--ntrain $ntrain \
	--save_dir "${save_dir}/hellaswag/${ntrain}-shot/" \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function
done





# --------------------------------------------------------------------------------
#				English to Indic MT
#---------------------------------------------------------------------------------

# FLORES
for ntrain in ${ntrains[@]}; do
for lang in ${langs[@]}; do

echo "Evaluating ${ntrain}-shot FLORES from en-$lang..."
echo "Results will be stored at $save_dir/flores/en-$lang/$ntrain-shot/"
python3 -m eval.flores.$run_eval \
	--ntrain $ntrain \
	--save_dir "${save_dir}/flores/${ntrain}-shot/" \
	--src_lang en \
	--tgt_lang $lang \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function

echo "Evaluating ${ntrain}-shot FLORES from $lang-en..."
echo "Results will be stored at $save_dir/flores/$lang-en/$ntrain-shot/"
python3 -m eval.flores.$run_eval \
	--ntrain $ntrain \
	--save_dir "${save_dir}/flores/${ntrain}-shot/" \
	--src_lang $lang \
	--tgt_lang en \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function
done
done

# IN22-Gen
for ntrain in ${ntrains[@]}; do
for lang in ${langs[@]}; do

echo "Evaluating ${ntrain}-shot FLORES from en-$lang..."
echo "Results will be stored at $save_dir/in22-gen/en-$lang/$ntrain-shot/"
python3 -m eval.in22.$run_eval \
	--ntrain $ntrain \
	--save_dir "${save_dir}/in22-gen/${ntrain}-shot/" \
	--src_lang en \
	--tgt_lang $lang \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function

echo "Evaluating ${ntrain}-shot FLORES from $lang-en..."
echo "Results will be stored at $save_dir/in22-gen/$lang-en/$ntrain-shot/"
python3 -m eval.in22.$run_eval \
	--ntrain $ntrain \
	--save_dir "${save_dir}/in22-gen/${ntrain}-shot/" \
	--src_lang $lang \
	--tgt_lang en \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function
done
done





# --------------------------------------------------------------------------------
#					Indic NLG
#---------------------------------------------------------------------------------

# IndicQA no context
for ntrain in ${ntrains[@]}; do
for lang in ${langs[@]}; do

echo "Evaluating ${ntrain}-shot IndicQA with no context..."
echo "Results will be stored at ${save_dir}/indicqa/no-context/$ntrain-shot/"
python3 -m eval.indicqa.$run_eval \
	--ntrain $ntrain \
	--max_context_length 768 \
	--no_context \
	--save_dir "${save_dir}/indicqa/no-context/${ntrain}-shot/" \
	--lang $lang \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function
done
done

# IndicQA with context
for ntrain in ${ntrains[@]}; do
for lang in ${langs[@]}; do

echo "Evaluating ${ntrain}-shot IndicQA with context..."
echo "Results will be stored at ${save_dir}/indicqa/with-context/$ntrain-shot/"
python3 -m eval.indicqa.$run_eval \
	--ntrain $ntrain \
	--max_context_length 768 \
	--no_context \
	--save_dir "${save_dir}/indicqa/with-context/${ntrain}-shot/" \
	--lang $lang \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function
done
done

# Indic Headline
for ntrain in ${ntrains[@]}; do
for lang in ${langs[@]}; do

echo "Evaluating ${ntrain}-shot IndicHeadline..."
echo "Results will be stored at ${save_dir}/indicheadline/$ntrain-shot/"
python3 -m eval.indicheadline.$run_eval \
	--ntrain $ntrain \
	--max_context_length 512 \
	--save_dir "${save_dir}/indicheadline/${ntrain}-shot/" \
	--lang $lang \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function
done
done

# IndicWikiBio
for ntrain in ${ntrains[@]}; do
for lang in ${langs[@]}; do

echo "Evaluating ${ntrain}-shot IndicWikiBio..."
echo "Results will be stored at ${save_dir}/indicwikibio/$ntrain-shot/"
python3 -m eval.indicwikibio.$run_eval \
	--ntrain $ntrain \
	--max_context_length 512 \
	--save_dir "${save_dir}/indicwikibio/${ntrain}-shot/" \
	--lang $lang \
	--model_name_or_path $model_name_or_path \
	--tokenizer_name_or_path $model_name_or_path \
	--eval_batch_size $eval_batch_size \
	$chat_formatting_function
done
done
