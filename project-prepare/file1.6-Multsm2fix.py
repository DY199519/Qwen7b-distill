# Iterate through the models above to fill in answers that failed due to network errors

import os, json, time, datetime
from pathlib import Path

import httpx
from openai import OpenAI

###############################################################################
# 0. Model Interface Configuration
###############################################################################
models_config = {
    "gemini-2.5-flash-preview-04-17-thinking": {
        "api_key": "sk-VJrRRrYljSfcLQPKD2ocOw8NrKaFOPsTszZy1gb5qWJixq2Y",
        "base_url": "https://api.aigptapi.com/v1/"
    },
    "grok-3-beta": {
        "api_key": "sk-VJrRRrYljSfcLQPKD2ocOw8NrKaFOPsTszZy1gb5qWJixq2Y",
        "base_url": "https://api.aigptapi.com/v1/"
    },
    # "qwen3-235b-a22b": {
    #     "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
    #     "base_url": "https://api.vansai.cn/v1"
    # },
    "doubao-pro-256k": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "moonshot-v1-8k": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "deepseek-v3": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "hunyuan-turbo": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "qwen2.5-72b-instruct": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    }
}

###############################################################################
# 1. Input/Output Paths
###############################################################################
input_file  = r"D:\project\final_ans_multi_sm.json"
output_dir  = r"D:\project"

os.makedirs(output_dir, exist_ok=True)                 # Ensure directory exists
in_path  = Path(input_file)
if not in_path.exists():
    raise FileNotFoundError(f"Input file not found: {in_path}")

###############################################################################
# 2. Helper Functions
###############################################################################
def contains_error(text: str | None) -> bool:
    return isinstance(text, str) and "error" in text.lower()

def get_completion(client: OpenAI, model: str, prompt: str,
                   attempt: int = 1, max_attempts: int = 10) -> str:
    try:
        rsp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return rsp.choices[0].message.content
    except Exception as e:
        if attempt < max_attempts:
            print(f"⚠️ Request failed, retrying ({attempt}/{max_attempts})... Reason: {e}")
            time.sleep(2)
            return get_completion(client, model, prompt, attempt + 1, max_attempts)
        print(f"❌ Final failure: {e}")
        return f"ERROR: {e}"

###############################################################################
# 3. Read Input Data
###############################################################################
with in_path.open(encoding="utf-8") as f:
    raw_data = json.load(f)

if not isinstance(raw_data, list):
    raise ValueError("The top level of the input file should be an array, where each element is a model block.")

processed_all = []

###############################################################################
# 4. Iterate Through Models
###############################################################################
for model_block in raw_data:
    model_name = model_block.get("model_name", "").strip()
    print(f"\n{'='*60}\n🔍 Processing model: {model_name}\n{'='*60}")

    cfg = models_config.get(model_name)
    if not cfg:
        print(f"⚠️ No model configuration found, skipping: {model_name}")
        continue

    client = OpenAI(
        api_key    = cfg["api_key"],
        base_url   = cfg["base_url"],
        http_client= httpx.Client(verify=False)   # Example service certificate issue
    )

    for item in model_block.get("results", []):
        q = item.get("core_question", "").strip()
        ctx = item.get("context_info", "").strip()

        # basic_answer
        if contains_error(item.get("basic_answer")):
            prompt_basic = f'Please answer: "{q}".'
            print(f"🔄 basic_answer: {q}")
            item["basic_answer"] = get_completion(client, model_name, prompt_basic)

        # answer_with_context
        if contains_error(item.get("answer_with_context")):
            prompt_ctx = (
                f'Please answer: "{q}", using the following information, '
                f'answer in order of importance, with factors sorted by importance: {ctx}.'
            )
            print(f"🔄 answer_with_context: {q}")
            item["answer_with_context"] = get_completion(client, model_name, prompt_ctx)

    # Single model output
    model_fixed = Path(output_dir) / f"{model_name}_fixed.json"
    with model_fixed.open("w", encoding="utf-8") as f:
        json.dump(model_block, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved: {model_fixed}")

    processed_all.append(model_block)

###############################################################################
# 5. Consolidated Output
###############################################################################
all_fixed = Path(output_dir) / "finalmut_fixed_all.json"
with all_fixed.open("w", encoding="utf-8") as f:
    json.dump(processed_all, f, ensure_ascii=False, indent=4)

print(f"\n🎉 All completed, consolidated file: {all_fixed.resolve()}")
