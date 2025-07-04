import google.generativeai as genai
import os
import time
import random
import re
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json
from datasets import load_dataset, concatenate_datasets

# ===== Initialization =====
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# ===== Parameters =====
FIRST_LOAD_DATA = False
NUM_SAMPLES = 20        # number of data points to evaluate per prompt
MAX_STEPS = 5          # number of optimization steps
TOP_K = 5               # how many top prompts to include in meta prompt
USE_THREE_CLASSES = True   # if True: use positive/negative/neutral
VERBOSE = True          # if True: print detailed logs
STRICT_MODE = True      # if True: only accept clean 'positive', 'negative', 'neutral'
    
def load_30_newdata():
    dataset = load_dataset("imdb")
    neg_dataset = load_dataset("imdb", split="train[:200]")
    pos_dataset = load_dataset("imdb", split="train[-200:]")
    combined_dataset = concatenate_datasets([neg_dataset, pos_dataset])
    dataset = combined_dataset.shuffle(seed=42)
    dataset = dataset.shuffle(seed=42)
    sample_inputs = []
    for item in dataset:
        label = "positive" if item["label"] == 1 else "negative"
        text = item["text"][:400]  # trim long reviews for speed
        sample_inputs.append((text, label))
        if len(sample_inputs) >= NUM_SAMPLES:
            break
    return sample_inputs

def add_neutral(sample_inputs):
    if USE_THREE_CLASSES:
        sample_inputs.append(("It was fine. Nothing special but not bad.", "neutral"))
        sample_inputs.append(("Mediocre acting, average story.", "neutral"))
        sample_inputs.append(("It's an okay movie. Watchable but forgettable.", "neutral"))
        sample_inputs.append(("The plot was decent, nothing groundbreaking.", "neutral"))
        sample_inputs.append(("Average performance from the cast.", "neutral"))
        sample_inputs.append(("It's alright, meets expectations but doesn't exceed them.", "neutral"))
        sample_inputs.append(("Standard storyline with predictable outcomes.", "neutral"))
        sample_inputs.append(("The movie is fine for what it is.", "neutral"))
        sample_inputs.append(("Reasonably entertaining but not memorable.", "neutral"))
        sample_inputs.append(("It's a typical film in this genre.", "neutral"))
        # sample_inputs.append(("The acting was competent, script was ordinary.", "neutral"))
        # sample_inputs.append(("Neither impressed nor disappointed.", "neutral"))
        # sample_inputs.append(("It's watchable but won't blow you away.", "neutral"))
        # sample_inputs.append(("Solid but unremarkable filmmaking.", "neutral"))
        # sample_inputs.append(("The movie delivers what it promises, nothing more.", "neutral"))
    return sample_inputs
def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"數據已保存到 {filename}")

def load_from_json(filename):
    """從JSON文件讀取數據"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功從 {filename} 讀取數據")
        return data
    except FileNotFoundError:
        print(f"找不到文件: {filename}")
        return None
    except json.JSONDecodeError:
        print(f"文件格式錯誤: {filename}")
        return None




# ===== Accuracy Evaluation =====
def evaluate_prompt(prompt, inputs):
    correct = 0
    total = 0
    for text, true_label in inputs:
        query = f'{prompt}\n\nReview: "{text}"\nSentiment:'
        try:
            response = model.generate_content(query)
            cleaned_text = response.text.strip().lower()
            words = cleaned_text.split()
            allowed_words = {"neutral", "positive", "negative"} # 使用 set 查詢效能更好
            if words and words[0] in allowed_words:
                output = words[0]
            else:
                output = ""
            if VERBOSE:
                print(f"\n[REVIEW] {text}\n[EXPECTED] {true_label} | [LLM OUTPUT] {output}")
            if STRICT_MODE:
                matched = true_label in output.split()
            else:
                matched = true_label in output
            if matched:
                correct += 1
            total += 1
        except Exception as e:
            print(f"⚠️ Gemini Error: {e}")
            continue
    return correct / total if total > 0 else 0.0

# ===== Build Meta Prompt =====
def build_meta_prompt(history, top_k, ascending):
    intro = """
    You are optimizing prompts for sentiment classification of movie reviews.
    The goal is to write better prompts that help correctly classify reviews as positive, negative, or neutral.
    Below are previous prompts and their scores (accuracy out of 1.0):
    """
    if ascending:
        history = sorted(history, key=lambda x: -x[1])
    else:
        history = history[-top_k-1:]
    meta_lines = [f'Attempt {i+1}: Prompt: """{prompt}""", Score: {score:.2f}' for i, (prompt, score) in enumerate(history[:top_k])]
    guidance = """
    Suggest a new and improved prompt for the task above.
    Only output the prompt using this format:
    Prompt: "<your prompt here>"
    Do not include any other explanation.
    """
    return intro + "\n" + "\n".join(meta_lines) + guidance

# ===== Extract Prompt from Output =====
def extract_prompt(text):
    match = re.findall(r'Prompt:\s*"(.*?)"', text, re.DOTALL)
    return match[-1].strip() if match else None

# ===== Plot Accuracy Over Time =====
def plot_prompt_scores(history, filename):
    attempts = list(range(1, len(history) + 1))
    scores = [s for _, s in history]
    plt.figure(figsize=(10, 6))
    plt.plot(attempts, scores, marker='o')
    plt.title("Prompt Optimization Progress")
    plt.xlabel("Attempt")
    plt.ylabel("Accuracy (0-1.0)")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# ===== Optimization Loop =====
def run_prompt_optimization(sample_inputs):
    history = [("Just write how you feel about it.", 0.5)]  # bad initial prompt
    for step in range(MAX_STEPS):
        print(f"\n=====================\n🚀 Step {step+1} Starting...")
        meta_prompt = build_meta_prompt(history, TOP_K, ascending=True)
        print(f"\n\n{meta_prompt}\n\n")
        try:
            response = model.generate_content(meta_prompt).text.strip()
            print(response)
            prompt = extract_prompt(response)
            print(prompt)
            if prompt is None:
                print("⚠️ Failed to extract prompt. Skipping.")
                continue
            print(f"\n🔧 Using New Prompt: {prompt}")
            score = evaluate_prompt(prompt, sample_inputs)
            print(f"✅ Accuracy: {score:.2f}")
            history.append((prompt, score))
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            continue
        time.sleep(1)
    plot_prompt_scores(history, "prompt-optimization-v3.png")

# ===== Run =====
if __name__ == "__main__":
    if FIRST_LOAD_DATA: 
        sample_inputs = load_30_newdata()
        if USE_THREE_CLASSES:
            add_neutral(sample_inputs)
        save_to_json(sample_inputs, "imdb_db.json")
    else:
        sample_inputs = load_from_json("imdb_db.json")

    run_prompt_optimization(sample_inputs)