import google.generativeai as genai
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import time
import random
from dotenv import load_dotenv

# 初始化 Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# 模擬任務樣本：情緒分類
sample_inputs = [
    ("I love this movie!", "positive"),
    ("This is terrible.", "negative"),
    ("It was okay, not great but not bad either.", "neutral")
]

# 模擬 black-box 評估器：計算 prompt 對情緒分類任務的效果
def simulate_prompt_accuracy(prompt):
    score = 0
    for text, true_label in sample_inputs:
        if "love" in text and "positive" in prompt:
            score += 1
        elif "terrible" in text and "negative" in prompt:
            score += 1
        elif "okay" in text and "neutral" in prompt:
            score += 1
    return score + random.uniform(-0.3, 0.3)  # 加入隨機擾動模擬 noise

# 生成 meta-prompt，供 Gemini 學習改善 prompt
def build_prompt_meta_prompt(history, top_k, ascending):
    intro = """
You are designing prompts to improve a sentiment classification task.
The goal is to maximize classification accuracy on the following sentences:

1. "I love this movie!" → positive
2. "This is terrible." → negative
3. "It was okay, not great but not bad either." → neutral

Here are some previously used prompts and their scores (out of 3.0):
"""
    if ascending:
        history = sorted(history, key=lambda x: -x[1])
    else:
        history = history[-top_k-1:]
    meta_lines = [
        f'Attempt No. {i+1}: Prompt: """{prompt}""", Score: {score:.2f}'
        for i, (prompt, score) in enumerate(history[:top_k])
    ]
    guidance = """
Suggest a new improved prompt for the task above.
Only output the prompt in this format at the end:
Prompt: \"\"\"<your prompt here>\"\"\"
Do not include any other text or examples.
"""
    return intro + "\n" + "\n".join(meta_lines) + "\n" + guidance

# 從 Gemini 回傳中解析 prompt
def extract_prompt(text):
    match = re.findall(r'Prompt:\s*"""(.*?)"""', text, re.DOTALL)
    return match[-1].strip() if match else None

# 繪製分數變化圖
def plot_prompt_scores(history, filename):
    attempts = range(1, len(history) + 1)
    scores = [x[1] for x in history]
    plt.figure(figsize=(10, 6))
    plt.plot(attempts, scores, marker='o')
    plt.title("Prompt Optimization Progress")
    plt.xlabel("Attempt")
    plt.ylabel("Simulated Accuracy (out of 3)")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# 主程式：執行多輪 Prompt 優化
def run_prompt_optimization(model, steps=10, top_k=5, ascending=True, filename="prompt-optimization.png"):
    history = []
    for step in range(steps):
        print(f"\n=== Step {step+1} ===\n")
        meta_prompt = build_prompt_meta_prompt(history, top_k, ascending)
        response = model.generate_content(meta_prompt)
        output = response.text.strip()
        print(f"\nLLM Response:\n{output}")
        prompt = extract_prompt(output)
        if prompt is None:
            print("⚠️ Failed to extract prompt. Skipping...")
            continue
        score = simulate_prompt_accuracy(prompt)
        print(f"✅ Prompt Score: {score:.2f}")
        history.append((prompt, score))
        time.sleep(1)  # 給 Gemini 一些時間
    plot_prompt_scores(history, filename)

# 執行
if __name__ == "__main__":
    run_prompt_optimization(
        model=model,
        steps=10,
        top_k=5,
        ascending=True,
        filename="prompt-optimization.png"
    )
