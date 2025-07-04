import google.generativeai as genai
import numpy as np
import math
import re
import time
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# 評分函數（黑箱）
def objective_function(x):
    A = 10
    noise = np.random.normal(0, 1)
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * math.pi * xi)) for xi in x]) + noise

# 格式化 solution-score pair 作為 meta-prompt
def build_meta_prompt(history, top_k, ascending):
    meta_prompt1 = """
    You are optimizing a 3-dimensional real-valued vector to minimize a black-box function. Lower score is better.
    Previous solutions listed below:
    """
    if ascending:
        history = sorted(history, key=lambda x: x[1])
    else:
        history = history[-top_k-1:]
    i=1
    lines = []
    for x, score in history[:top_k]:
        lines.append(f"Attempt No. {i},  solution: {x}, Score: {score:.2f}")
        i += 1
    meta_prompt2 = """
    Suggest a new solution vector (3 real numbers).
    Try suggesting a best solution. You may observe the best value in history and attempt to do better.(Lower score is better)
    Please follow this exact output format at the end of your response:
    For example: Solution: [x1, x2, x3]
    Do not include any other numbers outside of this format.
    """
    return meta_prompt1 + "\n" + "\n    ".join(lines) + "\n" + meta_prompt2

def parse_last_solution(response_text):
    # 使用 re.findall() 找出所有符合樣板的捕獲組內容
    # 樣板 r"Solution:\s*\[([^\]]+)\]" 中的括號 (...) 是一個捕獲組，
    # re.findall() 會返回所有捕獲組匹配到的字串列表。
    pattern = r"Solution:\s*\[([^\]]+)\]"
    matches = re.findall(pattern, response_text)
    if not matches:
        return None
    last_match_content = matches[-1]
    try:
        numbers = [float(num.strip()) for num in last_match_content.split(",")]
        if len(numbers) == 3:
            return numbers
    except ValueError:
        pass
    return None

def draw_plt(history,name):
    print("\n=====================\nOptimization finished. Generating plot...")
    # attempts 是 X 軸 (1, 2, 3, ...)
    attempts = range(1, len(history) + 1)
    # scores 是 Y 軸，從 history 中取出每個元組的第二個元素 (分數)
    scores = [item[1] for item in history]
    plt.figure(figsize=(10, 6)) # 設定圖表大小，讓它更清晰
    plt.plot(attempts, scores, marker='o', linestyle='-', color='b')
    plt.title(f'Optimization History', fontsize=16)
    plt.xlabel('Attempt Number', fontsize=12)
    plt.ylabel('Evaluated Score (Lower is Better)', fontsize=12)
    plt.xticks(attempts) # 確保X軸刻度為整數
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) # 加入網格線
    plt.savefig(name)
    plt.show()

def run_optimization(k, ascending,name):
    # 主要 loop
    history = []
    for step in range(10):
        print("\n=====================\n")
        print(f"step {step+1}: \n")
        prompt = build_meta_prompt(history, k, ascending)
        print(f"\nPrompt right now:\n{prompt}\nEnd of prompt\n")    
        response = model.generate_content(prompt)
        text = response.text.strip()
        print(f"\n🔁 Step {step+1}: Gemini Answer\n{text}")
        print("\n=========\n")

        solution = parse_last_solution(text)
        print(f"new_parse_solution:{solution}")
        if solution is None:
            print("⚠️ Failed to parse solution. Skipping.")
            continue

        score = objective_function(solution)
        print(f"✅ Evaluated Score: {score:.2f}")
        history.append((solution, score))
        time.sleep(1)  # 給 API 休息一下
    draw_plt(history, name)

if __name__ == "__main__":
    run_optimization(k=3, ascending=False,name ="k=3-numeric")
