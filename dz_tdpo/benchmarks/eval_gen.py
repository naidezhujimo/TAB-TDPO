import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import evaluate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dz_tdpo.config import TDPODKLConfig
from dz_tdpo.model import TemporalCausalLM
from dz_tdpo.data.dataset import TemporalPreferenceDataset
from dz_tdpo.data.msc import msc_to_temporal_preference

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Generation Quality & PPL")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()

def compute_f1_token(prediction, truth):
    pred_tokens = prediction.lower().split()
    truth_tokens = truth.lower().split()
    if not pred_tokens or not truth_tokens: return int(pred_tokens == truth_tokens)
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    
    p = 1.0 * num_same / len(pred_tokens)
    r = 1.0 * num_same / len(truth_tokens)
    return 100 * (2 * p * r) / (p + r)

def evaluate_model(model, tokenizer, dataloader, device, metrics_dict):
    model.eval()
    predictions = []
    references = []
    f1_scores = []
    total_nll = 0
    total_tokens = 0
    
    print("🚀 Starting Evaluation loop...")
    
    for batch in tqdm(dataloader, desc="Eval"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ref_texts = batch['ref_texts']
        
        with torch.no_grad():
            if hasattr(model, "model"):
                outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            
            nll = outputs.loss * input_ids.shape[1]
            total_nll += nll.item()
            total_tokens += input_ids.shape[1]

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
        
        input_len = input_ids.shape[1]
        for i, out_seq in enumerate(gen_out):
            gen_seq = out_seq[input_len:]
            pred_text = tokenizer.decode(gen_seq, skip_special_tokens=True).strip()
            ref_text = ref_texts[i].strip()
            
            if not pred_text: pred_text = "."
            
            predictions.append(pred_text)
            references.append(ref_text)
            f1_scores.append(compute_f1_token(pred_text, ref_text))

    print("📊 Computing Metrics...")
    
    refs_list = [[r] for r in references]
    res_bleu = metrics_dict['bleu'].compute(predictions=predictions, references=refs_list)
    res_rouge = metrics_dict['rouge'].compute(predictions=predictions, references=references)
    
    print("⏳ Computing BERTScore (DistilBERT)...")
    res_bert = metrics_dict['bert'].compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
    
    return {
        "BLEU": res_bleu['score'],
        "ROUGE-L": res_rouge['rougeL'] * 100,
        "F1": np.mean(f1_scores),
        "BERT-F1": np.mean(res_bert['f1']) * 100
    }

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("📥 Loading Evaluate metrics...")
    metrics_dict = {
        'bleu': evaluate.load("sacrebleu"),
        'rouge': evaluate.load("rouge"),
        'bert': evaluate.load("bertscore")
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({'additional_special_tokens': ["<|user|>", "<|assistant|>", "<|end|>"]})

    print(f"📦 Loading MSC Validation Data (N={args.num_samples})...")

    val_container = msc_to_temporal_preference(tokenizer, data_dir=args.data_dir, split="validation", sessions_per_sample=4)

    subset_samples = val_container.samples[:args.num_samples]

    dummy_config = TDPODKLConfig()
    val_dataset = TemporalPreferenceDataset(subset_samples, tokenizer, dummy_config)
    
    def gen_collate_fn(batch):
        ctx_list = [b['input_ids'] for b in batch]
        padded_input = pad_sequence(ctx_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = (padded_input != tokenizer.pad_token_id).long()
        
        ref_texts = []
        for b in batch:
            valid = (b['masks_w'] != -100)
            reply_ids = b['labels_w'][valid]
            ref_texts.append(tokenizer.decode(reply_ids, skip_special_tokens=True))
            
        return {
            'input_ids': padded_input,
            'attention_mask': attention_mask,
            'ref_texts': ref_texts
        }

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=gen_collate_fn)

    print("🧠 Loading Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    base_model.resize_token_embeddings(len(tokenizer))
    
    config = TDPODKLConfig(model_name=args.model_name_or_path, use_temporal_bias=True)
    model = TemporalCausalLM(base_model, config, device)
    
    model.load_weights(args.ckpt_path)
    
    res = evaluate_model(model, tokenizer, val_loader, device, metrics_dict)

    print("\n" + "="*60)
    print(f"🏆 Results (N={args.num_samples})")
    print("-" * 60)
    for k, v in res.items():
        print(f"{k:<10} | {v:.4f}")
    print("="*60)

if __name__ == "__main__":

    main()
