import random
import math
import numpy as np
import pandas as pd

# Define a black-box objective function (e.g., noisy Rastrigin function for optimization)
def objective_function(x):
    """A black-box function to minimize. Adding noise to simulate real-world uncertainty."""
    A = 10
    noise = np.random.normal(0, 1)  # add noise to simulate real-world conditions
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * math.pi * xi)) for xi in x]) + noise

# Simulated "LLM" optimizer that proposes new solutions based on meta-prompt (simulated)
def generate_solutions(meta_prompt, n=5):
    """
    Simulates generating new solutions from the LLM.
    The meta_prompt contains past solutions and their scores.
    """
    top_solutions = sorted(meta_prompt, key=lambda x: x[1])[:3]
    new_solutions = []
    for _ in range(n):
        base = random.choice(top_solutions)[0]
        new = [xi + np.random.normal(0, 1) for xi in base]
        new_solutions.append(new)
    return new_solutions

# Meta-optimizer framework
def OPRO_optimizer(objective_fn, dim=3, steps=10, initial_population=5):
    """
    Optimization by Prompting (OPRO) framework simulation.
    """
    # Initialize random solutions
    history = [([random.uniform(-5, 5) for _ in range(dim)], None) for _ in range(initial_population)]
    print(history, "\n======\n")
    # Evaluate initial solutions
    for i in range(len(history)):
        history[i] = (history[i][0], objective_fn(history[i][0]))
        print(history[i][0],objective_fn(history[i][0]),"\n======\n")
    for step in range(steps):
        # Simulate LLM optimizer generating new solutions
        generated = generate_solutions(history)
        
        # Evaluate new solutions
        scored = [(sol, objective_fn(sol)) for sol in generated]
        print(scored, "\n===\n")
        # Update history (meta-prompt)
        history.extend(scored)
        history = sorted(history, key=lambda x: x[1])[:20]  # Keep top-20 solutions only

    # Return top solution
    return sorted(history, key=lambda x: x[1])[0], history

best_solution, solution_history = OPRO_optimizer(objective_function)
df = pd.DataFrame([{"x": sol, "score": score} for sol, score in solution_history])
print(df)