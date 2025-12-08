import json
import torch
import random
import os
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dz_tdpo.data.msc import msc_to_temporal_preference
from dz_tdpo.data.ultrachat import build_ultrachat_dataset

def save_dataset_to_json(dataset, output_file):
    data_to_save = []
    if isinstance(dataset, list):
        iterator = dataset
    else:
        iterator = dataset.samples

    print(f"Saving {len(iterator)} samples to {output_file}...")
    for i in tqdm(range(len(dataset))):
        s = dataset[i]
        data_to_save.append({
            "context_ids": s.context_ids.tolist(),
            "chosen_reply_ids": s.chosen_reply_ids.tolist(),
            "rejected_reply_ids": s.rejected_reply_ids.tolist(),
            "context_turns": s.context_turns,
            "turn_id": s.turn_id,
            "total_turns": s.total_turns
        })
    
    with open(output_file, "w") as f:
        json.dump(data_to_save, f)
    print(f"Saving Finish: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msc_data_dir", type=str, required=True, help="Path to raw MSC dataset")
    parser.add_argument("--ultrachat_data_dir", type=str, default=None, help="Path to UltraChat")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ["<|user|>", "<|assistant|>", "<|end|>"]})
    
    print("Freezing MSC Data...")
    msc_ds = msc_to_temporal_preference(
        tokenizer,
        data_dir=args.msc_data_dir,
        split="validation",
        sessions_per_sample=4, 
        neg_distance=5
    )

    msc_samples = msc_ds.samples[:500] 
    save_dataset_to_json(msc_samples, "fixed_msc_test.json")

    if args.ultrachat_data_dir:
        print("\nFreezing UltraChat Data...")
        uc_samples = build_ultrachat_dataset(tokenizer, data_dir=args.ultrachat_data_dir, num_samples=300, split="test_sft")
        save_dataset_to_json(uc_samples, os.path.join(args.output_dir, "fixed_ood_test.json"))

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()