import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, DPOTrainer, ORPOTrainer, ORPOConfig, DPOConfig
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchDPOTrainer

HF_TOKEN = 'HF_API_KEY'

EOS_TOKEN = "<|endoftext|>"

ALPACA_PROMPT = """Below is an instruction describing a task, paired with input providing the article's content and its narrative. Write a detailed explanation (at least 80 words) in {language_1} that justifies why the given narrative applies to the article.

### Instruction:
{instruction}

### Input:
Document:
{document}

The main narrative of the article is "{narrative}".

Task:
Analyze the document and provide a detailed explanation (at least 80 words) in {language_2} showing how the narrative is reflected in the document.

### Response:
"""

MODEL_MAPPING = {
    "meta": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "mistral": "unsloth/mistral-7b-v0.3-bnb-4bit",
    "phi": "unsloth/Phi-3.5-mini-instruct",
    "gemma": "unsloth/gemma-2-9b-bnb-4bit",
    "llama": "unsloth/Llama-3.2-3B-bnb-4bit",
    "default_sft": "irasalsabila/sft_lora_model_adapters"
}

LANGUAGE_MAPPING = {
    "EN": "English",
    "BG": "Bulgarian",
    "HI": "Hindi",
    "PT": "Portuguese"
}

def replace_abbreviations(df, mapping):
    """Replaces abbreviations in the categories and subcategories."""
    df['categories'] = df['categories'].replace(mapping, regex=True)
    df['subcategories'] = df['subcategories'].replace(mapping, regex=True)
    return df

def formatting_prompts_func(examples, mode="sft"):
    """Formats the dataset for SFT, ORPO, or DPO modes."""
    languages = [LANGUAGE_MAPPING.get(lang, "English") for lang in examples['language']]
    documents = examples['paragraph']
    categories = examples['categories']
    subcategories = examples['subcategories']

    if mode in ["orpo", "dpo"]:
        prompts = []
        for doc, category, subcategory, lang in zip(documents, categories, subcategories, languages):
            narrative = subcategory if subcategory and subcategory.lower() != 'none' else category
            instruction = f"Provide an explanation on why the article reflects the given narrative in {lang}."
            prompts.append(ALPACA_PROMPT.format(language_1=lang, instruction=instruction, document=doc, narrative=narrative, language_2=lang))
        examples["prompt"] = prompts
        examples["chosen"] = examples['explanation']
        examples["rejected"] = examples['generated']
        return examples

    elif mode == "sft":
        outputs = examples.get("explanation", [""] * len(documents))
        texts = []
        for doc, category, subcategory, lang, output in zip(documents, categories, subcategories, languages, outputs):
            narrative = subcategory if subcategory and subcategory.lower() != 'none' else category
            instruction = f"Provide an explanation on why the article reflects the given narrative in {lang}."
            texts.append(ALPACA_PROMPT.format(language_1=lang, instruction=instruction, document=doc, narrative=narrative, language_2=lang, output=output) + EOS_TOKEN)
        return {"text": texts}

    else:
        raise ValueError("Invalid mode specified. Choose 'sft', 'orpo', or 'dpo'.")

def preprocess_data(train_file, dev_file, abbreviation_mapping, mode="sft"):
    """Preprocesses the training and development datasets."""
    df_train = pd.read_csv(train_file)
    df_dev = pd.read_csv(dev_file)

    df_train = replace_abbreviations(df_train, abbreviation_mapping)
    df_dev = replace_abbreviations(df_dev, abbreviation_mapping)

    train_dataset = Dataset.from_pandas(df_train).map(lambda x: formatting_prompts_func(x, mode=mode), batched=True)
    dev_dataset = Dataset.from_pandas(df_dev).map(lambda x: formatting_prompts_func(x, mode="sft"), batched=True)

    return train_dataset, dev_dataset

def setup_model(model, max_seq_length=2408):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
        token=HF_TOKEN
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )

    return model, tokenizer

def train_model(mode, dataset, model, tokenizer, output_dir, max_seq_length=2048):
    if mode == "sft":
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=60,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                output_dir=output_dir
            )
        )
    elif mode == "orpo":
        PatchDPOTrainer()
        trainer = ORPOTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=ORPOConfig(max_length=max_seq_length, max_prompt_length=max_seq_length // 2, max_completion_length=max_seq_length // 2,
                            per_device_train_batch_size=2, gradient_accumulation_steps=4, beta=0.1, logging_steps=1,
                            optim="adamw_8bit", max_steps=30, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
                            output_dir=output_dir)
        )
    elif mode == "dpo":
        PatchDPOTrainer()
        trainer = DPOTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=DPOConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_ratio=0.1,
                           num_train_epochs=2, learning_rate=5e-6, max_steps=200, fp16=not is_bfloat16_supported(),
                           bf16=is_bfloat16_supported(), output_dir=output_dir)
        )
    else:
        raise ValueError("Invalid mode specified.")

    print(f"Starting {mode.upper()} training...")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"{mode.upper()} training complete and model saved to {output_dir}.")

def plot_training_loss(trainer, output_file):
    loss_logs = trainer.state.log_history
    steps = [log["step"] for log in loss_logs if "loss" in log]
    loss_values = [log["loss"] for log in loss_logs if "loss" in log]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_values, marker='o', linestyle='-', label='Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Steps')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def generate_responses(dataset, model, tokenizer, output_file='./predictions.txt', max_retries=3):
    responses = []

    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_dict('records')

    for sample in dataset:
        FastLanguageModel.for_inference(model)
        retries = 0
        valid_response = False
        language = LANGUAGE_MAPPING.get(sample.get('language', 'EN'), 'English')
        instruction = "Analyze the document and explain how the given narrative applies."
        subcategory = sample.get('subcategories', '')
        category = sample.get('categories', '')
        narrative = subcategory if subcategory and subcategory.lower() != 'none' else category

        while not valid_response and retries < max_retries:
            retries += 1
            prompt = ALPACA_PROMPT.format(language_1=language, instruction=instruction, document=sample['paragraph'], narrative=narrative, language_2=language)
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=0.7 + (0.05 * retries), eos_token_id=tokenizer.eos_token_id)
            response = tokenizer.batch_decode(outputs)[0].strip().replace(EOS_TOKEN, '')
            if len(response.split()) >= 20:
                valid_response = True

        responses.append({"article_id": sample['filename'], "generated": response})

    df_predictions = pd.DataFrame(responses)
    df_predictions.to_csv(output_file, sep='\t', index=False, header=False, encoding='utf-8')
    print(f"Predictions saved: {output_file}")
    return df_predictions

def save_model_and_tokenizer(model, tokenizer, output_dir, save_method="default", token=None):
    """Saves the model and tokenizer locally and optionally pushes to Hugging Face."""
    save_path = os.path.join(output_dir, "model")
    os.makedirs(save_path, exist_ok=True)

    if save_method == "lora":
        model.save_pretrained_merged(save_path, tokenizer, save_method="lora")
        if token:
            model.push_to_hub_merged(f"irasalsabila/{output_dir}_adapters", tokenizer, save_method="lora", token=token)
    else:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    print(f"Model and tokenizer saved to: {save_path}")

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Train and evaluate a language model for narrative extraction.")
    parser.add_argument('--train_file', type=str, required=True, help="Path to the training dataset CSV file.")
    parser.add_argument('--dev_file', type=str, required=True, help="Path to the development dataset CSV file.")
    parser.add_argument('--output_root', type=str, default="./outputs", help="Root directory to save model and results.")
    parser.add_argument('--model', type=str, required=True, choices=MODEL_MAPPING.keys(), help="Choose one of the available models.")
    parser.add_argument('--mode', type=str, default="sft", choices=["sft", "orpo", "dpo"], help="Mode: SFT, ORPO, or DPO.")
    parser.add_argument('--token', type=str, required=False, help="Hugging Face API token for model push.")
    args = parser.parse_args()

    train_file = args.train_file
    dev_file = args.dev_file

    model_name = MODEL_MAPPING[args.model]
    mode = args.mode

    abbreviation_mapping = {"URW": "Ukraine-Russia War", "CC": "Climate Change"}

    if mode in ["orpo", "dpo"]:
        model_name = MODEL_MAPPING["default_sft"]

    train_dataset, dev_dataset = preprocess_data(train_file, dev_file, abbreviation_mapping, mode=mode)
    
    model, tokenizer = setup_model(model_name)
    
    # Train the model based on the mode
    output_dir = os.path.join(args.output_root, mode)
    train_model(mode=mode, dataset=train_dataset, model=model, tokenizer=tokenizer, output_dir=output_dir, max_seq_length=2048)

    # Save the trained model and tokenizer
    save_model_and_tokenizer(model, tokenizer, output_dir, save_method="lora", token=args.token)

    # Generate predictions
    predictions_file = os.path.join(output_dir, f"{mode}_predictions.txt")
    predictions = generate_responses(dev_dataset, model, tokenizer, output_file=predictions_file)

    print(f"Predictions saved to: {predictions_file}")

    elapsed_time = time.time() - start_time
    print(f"Script completed in {elapsed_time // 3600:.0f}h {(elapsed_time % 3600) // 60:.0f}m {elapsed_time % 60:.0f}s.")


if __name__ == "__main__":
    main()

    # # SFT Mode
    # python narrative_extraction.py --train_file ./subtask_3_train.csv --dev_file ./subtask_3_dev.csv --output_root ./sft_outputs --model llama --mode sft

    # # DPO Mode
    # python narrative_extraction.py --train_file ./train_data_balanced.csv --dev_file ./subtask_3_dev.csv --output_root ./dpo_outputs --model default_sft --mode dpo --token "TOKEN_HF"

    # # ORPO Mode
    # python narrative_extraction.py --train_file ./train_data_balanced.csv --dev_file ./subtask_3_dev.csv --output_root ./orpo_outputs --model default_sft --mode orpo --token "TOKEN_HF"
