import argparse
import os
import random
from sklearn import metrics
import torch
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import time
import evaluate
from datasets import load_dataset
from eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    dynamic_import_function,
)

templates = {
    "with_context": (
        "Answer the following question based on the information in the given passage.",
        "Passage:",
        "Question:",
        "Answer:",
    ),
    "no_context": (
        "Answer the following question.",
        "Question:",
        "Answer:"),
}


def trim_context(context, max_context_length, tokenizer):
    tokenized_context = tokenizer.encode(context, add_special_tokens=False)
    if len(tokenized_context) > max_context_length:
        context = tokenizer.decode(
            tokenized_context[:max_context_length], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    return context


def main(args):
    random.seed(args.seed)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None


    if args.script == "native":
        dataset = load_dataset("ai4bharat/IndicQA", f"indicqa.{args.lang}")
    else:
        dataset = load_dataset("ai4bharat/IndicQA-romanized", f"indicqa.{args.lang}")

    dataset = dataset.map(lambda x: {"context": x["context"].strip()})
    dataset = dataset.map(lambda x: {"question": x["question"].strip()})
    test_data = dataset["test"]

    prompts = []
    for i, example in enumerate(test_data):
        dev_data = test_data.filter(lambda x: x["question"] != example["question"]).shuffle(args.seed)
        k = args.ntrain

        if args.no_context:
            prompt, q_template, a_template = templates["no_context"]
            p_template = ""
        else:
            prompt, p_template, q_template, a_template = templates["with_context"]

        train_prompt = [{"role":"system", "content":prompt}]
        if k > 0:
            exemplars = dev_data.select(range(k))
            for dev_example in exemplars:
                answer = (
                    "unanswerable" if dev_example["answers"]["text"][0] == "" else dev_example["answers"]["text"][0]
                )
                if args.no_context:
                    user_prompt = q_template + " " + dev_example["question"] + "\n"
                    assistant_prompt = a_template + " " + answer
                else:
                    user_prompt = p_template
                    + " "
                    + trim_context(dev_example["context"], args.max_context_length, tokenizer)
                    + "\n"
                    + q_template
                    + " "
                    + dev_example["question"]
                    + "\n"
                    assistant_prompt = a_template + " " + answer
                train_prompt.extend(
                    [{"role":"user", "content":user_prompt}, {"role":"assistant", "content":assistant_prompt}]
                )

        if args.no_context:
            user_prompt = q_template + " " + format(example["question"]) + "\n"
        else:
            user_prompt = (
                p_template
                + " "
                + format(trim_context(example["context"], args.max_context_length, tokenizer))
                + "\n"
                + q_template
                + " "
                + format(example["question"])
                + "\n"
            )
        assistant_prompt = a_template
        prompt_end = [{"role":"user", "content":user_prompt}, {"role":"assistant", "content":assistant_prompt}]
        
        prompt = train_prompt + prompt_end
        if args.use_chat_format:
            prompt = chat_formatting_function(prompt)
        else:
            prompt = "\n\n".join([x["content"] for x in prompt])

        prompts.append(prompt)

    new_line_token = tokenizer.encode("\n", add_special_tokens=False)[
        -1
    ]  # get the last token because the tokenizer may add space tokens at the start.
    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=50,
        batch_size=args.eval_batch_size,
        stop_id_sequences=None,
    )
    # remove unnecessary space
    outputs = [output.strip().split("\n")[0] for output in outputs]

    with open(os.path.join(args.save_dir, f"indicqa_{args.lang}_predictions.jsonl"), "w") as fout:
        for example, output in zip(test_data, outputs):
            example["prediction_text"] = output
            fout.write(json.dumps(example) + "\n")

    print("Calculating F1, EM ...")
    metric = evaluate.load("squad")

    predictions = [{"id": example["id"], "prediction_text": output} for example, output in zip(test_data, outputs)]
    references = [{"id": example["id"], "answers": example["answers"]} for example in test_data]
    for i in range(len(references)):
        if references[i]["answers"]["text"][0] == "":
            references[i]["answers"]["text"][0] = "unanswerable"

    metrics = metric.compute(predictions=predictions, references=references)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=1, help="number of examples to use for few-shot evaluation.")
    parser.add_argument(
        "--no_context", action="store_true", help="If given, we're evaluating a model without the gold context passage."
    )
    parser.add_argument(
        "--max_context_length", type=int, default=768, help="maximum number of tokens in the context passage."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang", type=str, default="hi", choices=["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
    )
    parser.add_argument("--script", type=str, default="native", choices=["native", "romanized"])
    parser.add_argument("--save_dir", type=str, default="results/mmlu/llama-7B/")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts.",
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.",
    )
    args = parser.parse_args()
    main(args)