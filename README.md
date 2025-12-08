
<div align="center">

# DZ-TDPO: Non-Destructive Temporal Alignment for Mutable State Tracking in Long-Context Dialogue

<a href="DZ-TDPO.pdf"><img src="https://img.shields.io/badge/Paper-PDF_Available-b31b1b?style=for-the-badge&logo=adobeacrobatreader" alt="Paper PDF"></a>
<a href="https://twitter.com/YourTwitterHandle"><img src="https://img.shields.io/badge/Twitter-Thread-1DA1F2?style=for-the-badge&logo=twitter" alt="Twitter"></a>
<a href="#"><img src="https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge" alt="License"></a>

**Yijun Liao**  
*Independent Researcher*

[**Read the Paper (PDF)**](DZ-TDPO.pdf) | [**Twitter Thread**](https://x.com/KitaLiao)

</div>

---

## 📢 News

- **[2025/12/5]** 🔥 **Baseline Repair** Evaluation Pipeline Released! Added eval_judge.py (LLM-as-a-Judge) and freeze_data.py.
- **[2025/12/2]** 💻 **Code Released!** The full training framework and evaluation benchmarks are now available.
- **[2025/12/2]** 🚀 **Paper Released!** We have uploaded the paper PDF directly to this repo.

---

## ⚡️ Abstract

In long-context dialogue systems, models suffer from **State Inertia**, where static constraints prevent resolving conflicts between evolving user intents (e.g., "I'm now Vegan") and established historical context. Standard alignment methods like DPO incur a massive **"Alignment Tax"** (perplexity explosion >100) when trying to force these updates.

We propose **DZ-TDPO**, a non-destructive alignment framework that synergizes:
1.  **Conflict-Aware Dynamic KL Constraints (TDPO-DKL)**: Optimization level adjustment.
2.  **Learnable Temporal Attention Bias (Dual-Zone Temporal Attention)**: Representation level filtering powered by semantic conflict detection.

**Result:** DZ-TDPO achieves **State-of-the-Art win rates (50.8% on Qwen2.5-7B)** on the Multi-Session Chat (MSC) dataset while maintaining robust zero-shot generalization and negligible perplexity overhead.

---

## 🌟 Key Results

### 1. SOTA Performance on Mutable State Tracking
DZ-TDPO significantly outperforms Standard DPO and SimPO on the MSC dataset, solving the "State Inertia" problem without destroying the model's general capabilities.

<div align="center">
  <img src="assets/results.png" width="80%">
</div>

| Method | Win Rate (MSC) | PPL (Validation) | Alignment Tax |
| :--- | :---: | :---: | :---: |
| **Standard DPO** | 45.8% | 102.3 💥 | High |
| **SimPO** | 46.4% | 101.2 | High |
| **DZ-TDPO (Ours)** | **55.4%** | **26.0** ✅ | **Negligible** |

> **Note:** On **Qwen2.5-7B**, DZ-TDPO achieves a near-perfect **50.8% Win Rate**.

### 2. Robustness (Needle-in-a-Haystack)
Does the temporal decay make the model "forgetful"? **No.**
Our "Non-Conflicting Needle-in-a-Haystack" test confirms that DZ-TDPO retains 100% retrieval accuracy for non-conflicting facts, demonstrating precise attention regulation.

---

## 🛠️ Methodology

<div align="center">
  <img src="assets/framework.png" width="90%">
</div>

DZ-TDPO introduces a **Conflict-Aware Adaptive Decay** mechanism.
*   **Dual-Zone Temporal Attention:** We map dialogue turns into a latent semantic space using SBERT. A learnable scalar $\lambda$ applies a temporal bias *only* when a semantic conflict is detected between the current instruction and history.
*   **System Prompt Shielding:** We enforce a hard constraint to ensure the "System Prompt" (Safety Constitution) is never decayed, ensuring safety amidst aggressive state updates.

---

## 📂 Project Structure

```text
DZ-TDPO/
├── dz_tdpo/               # Core implementation package
│   ├── config.py          # Unified Configuration (TDPO & SimPO)
│   ├── loss.py            # Loss functions (TDPO-DKL, SimPO)
│   ├── model.py           # TemporalCausalLM & Attention Bias mechanism
│   ├── trainer.py         # Custom Trainer implementation
│   ├── utils.py           # Metrics & Semantic helpers
│   └── data/              # Data processing pipelines
│       ├── dataset.py     # Torch Dataset wrappers
│       ├── msc.py         # Multi-Session Chat (MSC) loader
│       └── ultrachat.py   # UltraChat loader
├── benchmarks/            # Comprehensive Evaluation Suite
│   ├── eval_tab60.py      # The "TAB-60" Adversarial Benchmark
│   ├── eval_needle.py     # Needle-in-a-Haystack Robustness Test
│   ├── eval_ppl.py        # Perplexity & Alignment Tax Evaluation
│   ├── eval_safety.py     # Context Flooding & Jailbreak Defense Test
│   ├── eval_pingpong.py   # Rapid Intent Switching Test (State Flip-Flop)
│   ├── eval_RULER.py      # Long-context Retrieval Stress Test
│   ├── merge_adapter.py   # Utility: Merge Custom Weights into HF Base Model
│   ├── eval_gen.py        # Quantitative: Generation Quality (BLEU, ROUGE, BERTScore)
│   ├── eval_judge.py      # LLM-as-a-Judge (DeepSeek v3.2)
│   ├── freeze_data.py     # Generate fixed test sets for reproducibility
│   ├── eval_generation_universal.py # Batch generation script
├── baselines/
│   ├── train_baselines.py # Training DPO and SimPO
├── train.py               # Main unified training entry point
├── test_cpu_dryrun.py     # Architecture integrity verification script
└── requirements.txt       # Project dependencies
```

## 💻 Installation & Usage
1. Clone the repository:
```bash
git clone https://github.com/YourUsername/DZ-TDPO.git
cd DZ-TDPO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
Note: For GPU acceleration, we recommend installing flash-attn separately.

   
## 🚀 Training
We provide a unified training script train.py that supports both DZ-TDPO and SimPO.

1. Prepare Data
Download the Multi-Session Chat (MSC) dataset and place it in your data directory (e.g., ./data/msc).

2. Run Training (DZ-TDPO)
To train with Temporal Bias and Adaptive Decay (requires sentence-transformers):

```bash
python train.py \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --data_dir ./data/msc \
    --output_dir ./checkpoints/dz_tdpo \
    --use_temporal_bias \
    --use_adaptive_tau \
    --batch_size 2 \
    --epochs 4
```

3. Run Training (DPO/SimPO Baseline)
```bash
python baselines/train_baselines.py \
    --mode dpo(/simpo) \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --data_dir ./data/msc \
    --output_dir ./checkpoints/dpo(/simpo)
```

## 🔬 Reproducible Evaluation Pipeline

To ensure the reproducibility of the **Win Rate** results reported in Table 1, we provide a complete evaluation pipeline consisting of data freezing, batch generation, and LLM-as-a-Judge automation.

### 1. Prepare Fixed Test Set
To guarantee that all models are evaluated on the exact same prompts, we first freeze a subset of the validation data.

```bash
# Generates 'fixed_msc_test.json' and 'fixed_ood_test.json' in ./data/
python benchmarks/freeze_data.py \
    --msc_data_dir ./data/msc \
    --output_dir ./data \
    --model_name microsoft/Phi-3.5-mini-instruct
```

### 2. Generate Responses
Use the universal generation script to generate model responses.
For DZ-TDPO (Ours):
```bash
python benchmarks/eval_generation_universal.py \
    --ckpt_path ./checkpoints/dz_tdpo/final_model.pt \
    --base_model_path microsoft/Phi-3.5-mini-instruct \
    --data_path ./data/fixed_msc_test.json \
    --output_file results_dz_tdpo.json \
    --use_tab \
    --batch_size 1
```
Note: The --use_tab flag is critical to enable the Dual-Zone Temporal Attention mechanism during inference.

For Baselines (e.g., Standard DPO):
```bash
python benchmarks/eval_generation_universal.py \
    --ckpt_path ./checkpoints/dpo/final_model.pt \
    --base_model_path microsoft/Phi-3.5-mini-instruct \
    --data_path ./data/fixed_msc_test.json \
    --output_file results_baseline.json \
    --batch_size 8
```

### 3. Automated Judgment (LLM-as-a-Judge)
We employ DeepSeek-V3.2 (or GPT-5) as an impartial judge to compute the Win Rate. The script handles position bias mitigation (swapping) and concurrent requests automatically.
Setup API Key:
```bash
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
# Or for OpenAI: export OPENAI_API_KEY="sk-..."
```
Run Evaluation:
```bash
python benchmarks/eval_judge.py \
    --ours results_dz_tdpo.json \
    --baseline results_baseline.json \
    --output judge_verdict.json \
    --model deepseek-chat \
    --workers 10
```

## 📊 Evaluation
We provide a comprehensive suite of benchmarks to evaluate State Tracking, Robustness, and Safety.

### 0. Model Merging (Optional)
If you want to merge the learned temporal bias and weights into the base model for easier deployment (HuggingFace format):
```bash
python benchmarks/merge_adapter.py \
    --base_model_path microsoft/Phi-3.5-mini-instruct \
    --ckpt_path ./checkpoints/dz_tdpo/final_model.pt \
    --save_path ./merged_model

### 1. Qualitative Analysis (TAB-60 & PingPong)
*   **TAB-60**: Evaluates the model on 60 adversarial scenarios (e.g., rapid preference toggling, role reversal).
    ```bash
    python benchmarks/eval_tab60.py --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```
*   **PingPong Test**: Tests the model's stability under high-frequency state updates (e.g., Vegan <-> Meat eater every turn).
    ```bash
    python benchmarks/eval_pingpong.py --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```

### 2. Robustness (Needle & RULER)
*   **Needle-in-a-Haystack**: Checks if the model can retrieve non-conflicting facts from long contexts (2k-16k tokens).
    ```bash
    python benchmarks/eval_needle.py --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```
*   **RULER**: Stress tests long-context retrieval capabilities.
    ```bash
    python benchmarks/eval_RULER.py --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```

### 3. Quantitative Metrics (PPL & Generation)
*   **Perplexity (Alignment Tax)**:
    ```bash
    python benchmarks/eval_ppl.py --data_dir ./data/msc --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```
*   **Generation Quality (BLEU/ROUGE/BERTScore)**:
    ```bash
    python benchmarks/eval_gen.py --data_dir ./data/msc --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```

## 📜 Citation
If you find this work helpful, please consider citing:
```latex
@misc{liao2025dztdponondestructivetemporalalignment,
      title={DZ-TDPO: Non-Destructive Temporal Alignment for Mutable State Tracking in Long-Context Dialogue}, 
      author={Yijun Liao},
      year={2025},
      eprint={2512.03704},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.03704}, 
}
```
<div align="center">
</div>
