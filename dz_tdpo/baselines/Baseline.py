import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig, CPOTrainer, CPOConfig
from tqdm import tqdm
from datasets import Dataset as HFDataset

from dz_tdpo.data.msc import msc_to_temporal_preference

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["dpo", "simpo"], help="Training mode")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MSC dataset")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--output_dir", type=str, default="./baseline_checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|end|>'})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {'additional_special_tokens': ["<|user|>", "<|assistant|>", "<|end|>"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    print("Loading MSC dataset using your original pipeline...")
    train_dataset = msc_to_temporal_preference(
        tokenizer,
        data_dir=args.data_dir,
        split="train",
        sessions_per_sample=4,
        neg_distance=5
    )
    val_dataset = msc_to_temporal_preference(
        tokenizer,
        data_dir=args.data_dir,
        split="validation",
        sessions_per_sample=4,
        neg_distance=5
    )

    print(f"Loaded {len(train_dataset)} samples.")

    def convert_dataset_to_trl_list(custom_dataset, tokenizer):
        trl_data = []
        for sample in tqdm(custom_dataset.samples, desc="Converting to TRL format"):
            trl_data.append({
                "prompt": tokenizer.decode(sample.context_ids, skip_special_tokens=False),
                "chosen": tokenizer.decode(sample.chosen_reply_ids, skip_special_tokens=False),
                "rejected": tokenizer.decode(sample.rejected_reply_ids, skip_special_tokens=False)
            })
        return trl_data

    print("Formatting train data for TRL...")
    train_data_list = convert_dataset_to_trl_list(train_dataset, tokenizer)
    
    print("Formatting val data for TRL...")
    val_data_list = convert_dataset_to_trl_list(val_dataset, tokenizer)
    
    trl_train_dataset = HFDataset.from_list(train_data_list)
    trl_val_dataset = HFDataset.from_list(val_data_list)
    
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading Base Model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        attn_implementation="sdpa",
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))

    if args.mode == "dpo":
        print("Starting Standard DPO Training...")
        training_args = DPOConfig(
            output_dir=f"{args.output_dir}/dpo",
            beta=0.1,
            learning_rate=1e-6, 
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            num_train_epochs=6,
            logging_steps=4,
            save_steps=50,
            save_total_limit=2,
            max_length=2400,
            max_prompt_length=3000,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            bf16=True,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="rewards/accuracies",
        )
        
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=trl_train_dataset,
            eval_dataset=trl_val_dataset,
            processing_class=tokenizer,
        )

    elif args.mode == "simpo":
        print("Starting SimPO Training...")
        training_args = CPOConfig(
            output_dir=f"{args.output_dir}/simpo",
            loss_type="simpo",
            beta=2.0,
            simpo_gamma=1.4,
            learning_rate=5e-7,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=64,
            num_train_epochs=25,
            logging_steps=1,
            save_steps=50,
            max_length=2400,
            max_prompt_length=3000,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            bf16=True,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="rewards/accuracies",
        )
        
        trainer = CPOTrainer(
            model=model,
            args=training_args,
            train_dataset=trl_train_dataset,
            eval_dataset=trl_val_dataset,
            processing_class=tokenizer,
        )

    print(f"Start training with {len(trl_train_dataset)} samples...")
    trainer.train()
    trainer.save_model(f"{args.output_dir}/{args.mode}_final")
    print("Training Finished!")

if __name__ == "__main__":
    main()
