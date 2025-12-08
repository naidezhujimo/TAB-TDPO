import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os
from dataclasses import dataclass
from typing import List
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dz_tdpo.config import TDPODKLConfig


class TemporalAttentionBias(nn.Module):
    def __init__(self, config: TDPODKLConfig, max_positions: int = 32768, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.lambda_strength = nn.Parameter(torch.tensor(config.bias_lambda, dtype=dtype))
        self.register_buffer("tau_fixed", torch.tensor(config.tau_fixed, dtype=dtype))

class TemporalCausalLM_Gen(nn.Module):
    def __init__(self, model, config, device):
        super().__init__()
        self.model = model
        self.device = device
        self.config = config
        self.model_dtype = model.dtype if hasattr(model, 'dtype') else torch.float32
        self.original_forward = model.forward
        model.forward = self.forward_with_bias
        self.temporal_bias = None
        if config.use_temporal_bias:
            self.temporal_bias = TemporalAttentionBias(config, dtype=torch.float32).to(device)
        self.current_turn_boundaries = None

    def forward_with_bias(self, *args, **kwargs):
        def sanitize_outputs(outputs):
            if hasattr(outputs, 'logits') and outputs.logits.dim() == 4:
                if outputs.logits.shape[1] == 1: outputs.logits = outputs.logits.squeeze(1)
                elif outputs.logits.shape[2] == 1: outputs.logits = outputs.logits.squeeze(2)
            return outputs

        input_ids = kwargs.get('input_ids', args[0] if len(args) > 0 else None)
        current_mask = kwargs.get('attention_mask', args[1] if len(args) > 1 else None)
        
        if current_mask is None or self.temporal_bias is None or self.current_turn_boundaries is None:
            return sanitize_outputs(self.original_forward(*args, **kwargs))

        k_len = current_mask.shape[1]
        q_len = input_ids.shape[1] if input_ids is not None else k_len
        min_val = torch.finfo(self.model_dtype).min
        
        # Extended Mask
        if current_mask.dim() == 2:
            extended_mask = (1.0 - current_mask) * min_val
            extended_mask = extended_mask.to(dtype=self.model_dtype).unsqueeze(1).unsqueeze(1)
        else:
            extended_mask = current_mask.to(dtype=self.model_dtype)

        if q_len > 1:
            causal_mask = torch.full((q_len, k_len), min_val, dtype=self.model_dtype, device=self.device)
            cond = torch.triu(torch.ones((q_len, k_len), device=self.device, dtype=torch.bool), diagonal=1 + (k_len - q_len))
            causal_mask = causal_mask.masked_fill(~cond, 0.0).unsqueeze(0).unsqueeze(0)
            extended_mask = extended_mask + causal_mask

        # Bias Calculation
        token_turn_ids = torch.zeros(k_len, dtype=torch.long, device=self.device)
        boundaries = self.current_turn_boundaries
        if boundaries:
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i+1]
                if start < k_len:
                    token_turn_ids[start : min(end, k_len)] = i
            last_boundary = boundaries[-1]
            if k_len > last_boundary:
                token_turn_ids[last_boundary:] = len(boundaries) - 1

        q_ids = token_turn_ids[-q_len:].unsqueeze(1) 
        k_ids = token_turn_ids.unsqueeze(0)
        dist = (q_ids - k_ids).to(self.model_dtype)
        lambda_s = self.temporal_bias.lambda_strength.to(self.model_dtype)
        tau_f = self.temporal_bias.tau_fixed.to(self.model_dtype)
        
        valid_history = (dist > 0).to(self.model_dtype)
        is_decayable = (k_ids > 0).to(self.model_dtype)
        
        bias = -torch.abs(lambda_s) * (dist / tau_f) * valid_history * is_decayable
        bias = torch.clamp(bias, min=-100.0).unsqueeze(0).unsqueeze(0)
        final_mask = extended_mask + bias.to(self.model_dtype)

        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = final_mask
            return sanitize_outputs(self.original_forward(*args, **kwargs))
        else:
            new_args = list(args)
            new_args[1] = final_mask
            return sanitize_outputs(self.original_forward(*new_args, **kwargs))

    def generate(self, input_ids, turn_boundaries = None, **kwargs):
        self.current_turn_boundaries = turn_boundaries
        if 'attention_mask' in kwargs and kwargs['attention_mask'].dim() > 2: del kwargs['attention_mask']
        try:
            return self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            self.current_turn_boundaries = None

@dataclass
class SimpleSample:
    context_ids: torch.Tensor
    chosen_reply_ids: torch.Tensor
    context_turns: List[int]

class JsonDataset(Dataset):
    def __init__(self, json_path):
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.samples = []
        for d in data:
            self.samples.append(SimpleSample(
                context_ids=torch.tensor(d['context_ids'], dtype=torch.long),
                chosen_reply_ids=torch.tensor(d['chosen_reply_ids'], dtype=torch.long),
                context_turns=d['context_turns']
            ))
            
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to final_model.pt")
    parser.add_argument("--base_model_path", type=str, default="microsoft/Phi-3.5-mini-instruct", help="Base model HF path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to fixed_msc_test.json")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--use_tab", action="store_true", help="Enable Temporal Attention Bias")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.use_tab:
        real_batch_size = 1
        print(f"Mode: DZ-TDPO (TAB Enabled) | Force Batch Size = 1 for safety")
    else:
        real_batch_size = args.batch_size
        print(f"Mode: Standard DPO/SimPO | Parallel Generation (Batch Size = {real_batch_size})")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, padding_side='left')
    tokenizer.add_special_tokens({'additional_special_tokens': ["<|user|>", "<|assistant|>", "<|end|>"]})
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.use_tab:
        print("-> Loading DZ-TDPO Model...")
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        base_model.resize_token_embeddings(len(tokenizer))
        config = TDPODKLConfig(
            model_name=args.base_model_path, 
            use_temporal_bias=True,
            bias_lambda=0.5,
            tau=8.0
        )
        model = TemporalCausalLM_Gen(base_model, config, device)
        ckpt = torch.load(args.ckpt_path, map_location=device)
        state_dict = ckpt['policy_state_dict'] if 'policy_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("-> Loading Standard Model...")
        model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16, device_map="auto")
        if model.config.vocab_size != len(tokenizer): 
            print(f"Resize embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

    model.eval()
    
    dataset = JsonDataset(args.data_path)

    def collate_fn(batch):
        input_ids_list = [item.context_ids for item in batch]
        
        max_len = max(len(ids) for ids in input_ids_list)
        
        padded_input_ids = []
        attention_masks = []
        
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            padded_ids = torch.cat([
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long),
                ids
            ])

            mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(len(ids), dtype=torch.long)
            ])
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
            
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "raw_samples": batch
        }

    dataloader = DataLoader(dataset, batch_size=real_batch_size, collate_fn=collate_fn, shuffle=False)
    
    results = []
    print(f"Generating...")
    
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        raw_samples = batch["raw_samples"]
        
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": tokenizer.pad_token_id
        }
        
        if args.use_tab:
            gen_kwargs["turn_boundaries"] = raw_samples[0].context_turns
        else:
            gen_kwargs["attention_mask"] = attention_mask
        
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
            
        new_tokens = outputs[:, input_ids.shape[1]:]
        generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        for i, sample in enumerate(raw_samples):
            prompt = tokenizer.decode(sample.context_ids, skip_special_tokens=False)
            truth = tokenizer.decode(sample.chosen_reply_ids, skip_special_tokens=True)
            
            results.append({
                "prompt": prompt,
                "ground_truth": truth,
                "model_response": generated_texts[i]
            })

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done! Saved to {args.output_file}")

if __name__ == "__main__":
    main()