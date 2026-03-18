# STEM基准在SOTA大模型中的评估

### 🧑‍💻 User

what's stem benchmark in sota llm?

---

### 🤖 Assistant

In the context of State-of-the-Art (SOTA) Large Language Models (LLMs), a **STEM benchmark** is a standardized dataset or test used to evaluate a model's proficiency in **Science, Technology, Engineering, and Mathematics**. 

Unlike standard NLP tasks (like summarizing text or writing emails), STEM benchmarks test an LLM's capacity for **complex logic, multi-step mathematical reasoning, spatial understanding, and rigorous fact-checking.**

Here is a breakdown of the most critical STEM benchmarks used to evaluate SOTA models today, how the top models perform, and why these tests are so difficult.

---

### The Gold Standard STEM Benchmarks

Evaluating LLMs in STEM is generally broken down into three categories: Mathematics, Coding (Technology/Engineering), and Hard Science. 

<details>
<summary><strong>1. Mathematics Benchmarks (Click to expand)</strong></summary>

Math is the ultimate test of logical reasoning. It requires models to plan a path to a solution without making cascading errors.

*   `GSM8K` **(Grade School Math 8K):** A dataset of middle-school-level math word problems. It takes 2 to 8 steps to solve. SOTA models have largely saturated this benchmark, scoring well over $90\%$.
*   `MATH` **(Mathematics Aptitude Test of Heuristics):** A much harder dataset of 12,500 problems from high school math competitions (like AMC 10, AMC 12, and AIME). It covers algebra, geometry, number theory, and calculus. 
*   `AIME` **(American Invitational Mathematics Examination):** Highly complex competition math. Models are often evaluated on specific years (e.g., `AIME 2024`). OpenAI's `o1` model recently made massive leaps here.

</details>

<details>
<summary><strong>2. Coding & Software Engineering (Click to expand)</strong></summary>

Coding evaluates the "Technology and Engineering" aspects of STEM. It tests if an LLM can understand syntax, logic, and system architecture.

*   `HumanEval`: Created by OpenAI, this dataset contains 164 handwritten programming problems in Python. The model must write a function body given a docstring and signature.
*   `MBPP` **(Mostly Basic Python Problems):** Around 1,000 beginner-level Python programming tasks.
*   `SWE-bench`: A highly rigorous benchmark where an LLM is given an actual GitHub issue from a real-world open-source Python repository (like `scikit-learn` or `django`) and must write a pull request to fix the bug.

</details>

<details>
<summary><strong>3. Science & Advanced Knowledge (Click to expand)</strong></summary>

These benchmarks test domain-specific knowledge in physics, biology, chemistry, and engineering.

*   `MMLU` **(Massive Multitask Language Understanding) - STEM Subset:** While `MMLU` covers 57 subjects, the STEM subset focuses on college-level physics, chemistry, computer science, and engineering.
*   `GPQA` **(Google-Proof Q&A):** The current ultimate test for advanced science. It contains multiple-choice questions written by PhDs in biology, physics, and chemistry. The questions are so difficult that even expert PhDs with access to Google struggle to answer them quickly. 

</details>

---

### How SOTA Models Currently Perform

The landscape of SOTA models changes rapidly, but as of late 2024, the top contenders in STEM are **OpenAI's o1 / GPT-4o**, **Anthropic's Claude 3.5 Sonnet**, and **Google's Gemini 1.5 Pro**.

#### The Paradigm Shift: Test-Time Compute (OpenAI `o1`)
For a long time, models struggled with hard math (like the `MATH` or `AIME` benchmarks) because they tried to predict the answer token-by-token instantly. 

OpenAI's `o1` model introduced hidden "chain-of-thought" reasoning, allowing the model to "think" (use test-time compute) before answering. This resulted in a massive leap in STEM scores:
*   On `MATH`, `o1-preview` scores around **$85.5\%$** (compared to GPT-4o's ~$76\%$).
*   On `GPQA` (PhD-level science), `o1` scores **$78\%$**, surpassing human experts.

#### The Coding Champion: `Claude 3.5 Sonnet`
While `o1` dominates pure math, Anthropic's `Claude 3.5 Sonnet` is widely considered the current SOTA for applied technology and engineering (coding). 
*   It excels on `SWE-bench`, where it can navigate complex, multi-file codebases to patch real-world bugs better than almost any other model.

#### The Multimodal STEM Engine: `Gemini 1.5 Pro`
Google's `Gemini 1.5 Pro` is exceptionally strong in multimodal STEM tasks (e.g., looking at a diagram of a physics problem or reading a 100-page academic paper on chemistry) due to its massive 2-million token context window. It scores highly across `MMLU` and `MATH`.

---

### Why are STEM Benchmarks so hard for LLMs?

LLMs are inherently probabilistic text generators. They predict the next word. STEM, however, requires **deterministic logic**. 

#### 1. The "Cascading Error" Problem in Math
If an LLM writes an essay and uses a slightly suboptimal word, the essay is still fine. But in math, a single token error destroys the entire solution. 

For example, if evaluating the Gaussian integral:
$$ \int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} $$

If the LLM makes a slight hallucination during a step-by-step derivation and drops a negative sign, changing $e^{-x^2}$ to $e^{x^2}$, the integral suddenly diverges to infinity. The model cannot simply "smooth over" this error; the final answer will be completely wrong.

#### 2. Execution vs. Prediction
In coding benchmarks like `HumanEval`, the LLM isn't just graded on whether the code *looks* right; the code is actually executed against hidden unit tests. 

```python
def is_prime(n: int) -> bool:
    """Return True if n is prime, False otherwise."""
    # The LLM must generate perfect logic here.
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```
If the model hallucinates a `>=` instead of a `>`, the code fails the benchmark immediately.

### The Future of STEM Benchmarks

Because SOTA models are rapidly saturating current benchmarks (scoring >90% on `MMLU` and `GSM8K`), researchers are forced to create harder tests. The future of STEM benchmarking is shifting toward:
1.  **Agentic Environments:** Instead of answering a multiple-choice question, the LLM is given a virtual Ubuntu environment, a Jupyter notebook, and a goal (e.g., "Train a machine learning model on this CSV to achieve 80% accuracy").
2.  **Formal Verification:** Using languages like `Lean` or `Coq` where the LLM writes a mathematical proof, and a compiler mathematically guarantees whether the proof is flawlessly true or false.