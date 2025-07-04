# 🧠 LLM-Guided Black-Box Optimization with Meta-Prompting

A toy framework exploring how Large Language Models (LLMs) like Gemini can assist in **black-box optimization** through a novel technique called **meta-prompting**.

> Can a language model optimize its own prompt?
> Can it solve math functions and classification tasks… through text?

---

## 🌟 Features

* 🧪 **Black-box function optimization** using LLM-generated solutions (Rastrigin + noise)
* 💬 **Prompt optimization for text classification** (sentiment analysis on IMDB)
* 🔁 **Meta-prompting strategy**: iteratively guides LLM by feeding top-k prior results
* 📊 Visualization of performance across optimization steps
* 🧰 Support for Google Gemini API + dotenv-based API key management

---

## 📂 Project Structure

```
autoprompting/
├── opro_automl.py        # Black-box optimization on numeric functions
├── opro_autoprompt.py    # Prompt optimization for classification
├── opro_obj.py           # Meta-prompt formatting & black-box definition
├── opro_prompt.py        # IMDB sentiment classification optimization
├── agno_test.py          # Gemini + yFinance integration demo
├── imdb_db.json          # Sample database for prompt generation
├── *.png                 # Visualizations of optimization curves
```

---

## 🚀 Quickstart

### 1. Clone and Install

```bash
git clone https://github.com/blackwingedkite/LLM-Guided-Black-Box-Optimization-with-Meta-Prompting.git
cd LLM-Guided-Black-Box-Optimization-with-Meta-Prompting
pip install -r requirements.txt
```

> Or manually install key dependencies:

```bash
pip install google-generativeai numpy matplotlib datasets python-dotenv
```

---

### 2. Set up API Key

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

### 3. Run Examples

#### ➤ Optimize Math Function (Rastrigin):

```bash
python autoprompting/opro_automl.py
```

#### ➤ Prompt Optimization for Sentiment Classification:

```bash
python autoprompting/opro_autoprompt.py
```

#### ➤ Full IMDB Prompt Loop:

```bash
python autoprompting/opro_prompt.py
```

---

## 📈 Meta-Prompting Example

Meta-prompt structure fed into the LLM:

```
You are optimizing a 3D vector to minimize a black-box function. Lower is better.

Attempt No.1, solution: [1.2, 3.4, -2.1], Score: 15.23  
Attempt No.2, solution: [2.0, 0.1, 4.7], Score: 19.85  
...
```

LLM then proposes new candidates based on top `k` solutions.

---

## 🧠 Conceptual Inspiration

This project explores:

* Using LLMs beyond language → as an **optimizer**
* Turning optimization history into **meta-prompts**
* Simulating **AutoML**-like workflows with pure text interaction

---

## 📷 Sample Visualizations

![](autoprompting/k=10-numeric.png)
*Optimization progression using k=10 meta-prompting strategy*

---

## 🧃 Maintainer

Created by [柯柯（Ke Youqi）](https://github.com/blackwingedkite)
If you use this project or like the idea, feel free to ⭐ star or open an issue!
