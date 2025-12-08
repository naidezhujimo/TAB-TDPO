import argparse
import json
import os
import random
import time
import re
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_json_response(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 尝试去除 Markdown 代码块标记
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return None

def get_judge_prompt(question, answer_a, answer_b):
    system_prompt = "You are a helpful assistant that acts as an impartial judge to evaluate the quality of AI responses."
    user_prompt = f"""I want you to act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.

You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as usefulness, relevance, accuracy, depth, creativity, and level of detail.

[User Question]
{question}

[Assistant A]
{answer_a}

[Assistant B]
{answer_b}

Which is better?
Output your decision in a strict JSON format with two keys: "reason" and "winner".
The "winner" must be one of "A", "B", or "Tie".
"""
    return system_prompt, user_prompt

def call_judge_api(client, model_name, system_prompt, user_prompt, max_retries=5):
    for i in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=512,
                timeout=60
            )
            content = completion.choices[0].message.content
            parsed_json = parse_json_response(content)
            
            if parsed_json and "winner" in parsed_json:
                return parsed_json
            else:
                print(f"JSON Parse Error (Retry {i+1}): {content[:50]}...")
                time.sleep(1)
                
        except Exception as e:
            wait_time = 2 ** (i + 1)
            print(f"API Error (Retry {i+1}): {e}. Waiting {wait_time}s...")
            time.sleep(wait_time)
            
    return {"winner": "Error", "reason": "Failed API call after retries"}

def process_single_sample(client, model_name, item):
    i = item['id']
    prompt = item['prompt']
    resp_ours = item['resp_ours']
    resp_base = item['resp_base']

    flip_coin = random.random() < 0.5
    
    if flip_coin:
        ans_a, ans_b = resp_ours, resp_base
        ours_is_a = True
    else:
        ans_a, ans_b = resp_base, resp_ours
        ours_is_a = False

    sys_p, user_p = get_judge_prompt(prompt, ans_a, ans_b)
    
    evaluation = call_judge_api(client, model_name, sys_p, user_p)
    winner = evaluation.get("winner", "Tie")

    if winner == "Error":
        res_type = "error"
    elif winner == "Tie":
        res_type = "tie"
    elif (winner == "A" and ours_is_a) or (winner == "B" and not ours_is_a):
        res_type = "win"
    else:
        res_type = "loss"

    return {
        "id": i,
        "prompt": prompt,
        "ours_response": resp_ours,
        "baseline_response": resp_base,
        "judge_raw_output": evaluation,
        "result": res_type
    }

def main():
    parser = argparse.ArgumentParser(description="Automated LLM-as-a-Judge Evaluation Script")
    parser.add_argument("--ours", type=str, required=True, help="Path to your model's generation json")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline generation json")
    parser.add_argument("--output", type=str, default="judge_results.json", help="Output path for judgment results")
    
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek/OpenAI API Key")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com", help="API Base URL")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name (e.g., deepseek-chat, gpt-4o)")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent threads")
    
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("\nAPI Key missing! Please set --api_key or env var DEEPSEEK_API_KEY.\n")
    
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print(f"Loading files:\n  Ours: {args.ours}\n  Base: {args.baseline}")
    with open(args.ours, 'r', encoding='utf-8') as f:
        ours_data = json.load(f)
    with open(args.baseline, 'r', encoding='utf-8') as f:
        base_data = json.load(f)

    min_len = min(len(ours_data), len(base_data))
    tasks = []
    
    print("Verifying data alignment...")
    for i in range(min_len):
        p1 = ours_data[i].get('prompt', '')
        p2 = base_data[i].get('prompt', '')
        
        if p1[:50] != p2[:50]:
            print(f"Warning: Prompts might vary at index {i}!")
            
        tasks.append({
            "id": i,
            "prompt": p1,
            "resp_ours": ours_data[i].get('model_response', ''),
            "resp_base": base_data[i].get('model_response', '')
        })

    print(f"Data aligned. Starting evaluation of {len(tasks)} samples with {args.workers} workers.")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {
            executor.submit(process_single_sample, client, args.model, t): t['id'] 
            for t in tasks
        }
        
        for future in tqdm(as_completed(future_to_id), total=len(tasks), desc="Judging"):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Worker exception: {e}")

    results.sort(key=lambda x: x['id'])
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    stats = {"win": 0, "loss": 0, "tie": 0, "error": 0}
    for r in results:
        stats[r['result']] += 1
    
    total = len(results)
    valid_total = total - stats["error"]
    win_rate = (stats["win"] + 0.5 * stats["tie"]) / valid_total * 100 if valid_total > 0 else 0

    print("\n" + "="*40)
    print(f"📊 DZ-TDPO Evaluation Report")
    print(f"Model: {args.model}")
    print(f"Total Samples: {total}")
    print(f"----------------------------------------")
    print(f"🟢 Wins:   {stats['win']:<5} ({stats['win']/total:.1%})")
    print(f"⚪ Ties:   {stats['tie']:<5} ({stats['tie']/total:.1%})")
    print(f"🔴 Losses: {stats['loss']:<5} ({stats['loss']/total:.1%})")
    print(f"❌ Errors: {stats['error']:<5}")
    print(f"----------------------------------------")
    print(f"🏆 Final Win Rate: {win_rate:.2f}%")
    print("="*40)
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()