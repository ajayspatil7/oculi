"""
GSM8K Dataset Loading for MATS 10.0
====================================

Loads and preprocesses GSM8K math problems for sycophancy experiments.

Features:
- Filter for multi-step problems (≥3 calculation steps)
- Generate plausible wrong answers
- Structured problem format for experiment pipeline
"""

import re
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


@dataclass
class GSM8KProblem:
    """Structured GSM8K problem with correct and wrong answers."""
    question: str
    answer: str  # Numeric answer only
    full_solution: str  # Full solution text
    wrong_answer: str  # Plausible wrong answer
    nonsense_reason: str  # Fake reasoning for wrong answer
    n_steps: int  # Number of calculation steps
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "full_solution": self.full_solution,
            "wrong_answer": self.wrong_answer,
            "nonsense_reason": self.nonsense_reason,
            "n_steps": self.n_steps,
        }


def load_gsm8k(split: str = "test") -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset.
    
    Args:
        split: Dataset split ("train" or "test")
        
    Returns:
        List of problem dicts with 'question' and 'answer' fields
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library not installed. Run: pip install datasets")
    
    dataset = load_dataset("gsm8k", "main", split=split)
    
    problems = []
    for item in dataset:
        problems.append({
            "question": item["question"],
            "answer": item["answer"],  # Contains full solution + answer
        })
    
    print(f"✓ Loaded {len(problems)} GSM8K problems from '{split}' split")
    return problems


def extract_answer(solution: str) -> str:
    """
    Extract numeric answer from GSM8K solution.
    
    GSM8K format: "...#### 42" where 42 is the answer
    
    Args:
        solution: Full solution text
        
    Returns:
        Numeric answer as string
    """
    # Look for #### pattern
    match = re.search(r"####\s*(\d+(?:,\d{3})*(?:\.\d+)?)", solution)
    if match:
        # Remove commas from numbers like "1,234"
        return match.group(1).replace(",", "")
    
    # Fallback: look for last number
    numbers = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", solution)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return ""


def count_calculation_steps(solution: str) -> int:
    """
    Count calculation steps in a GSM8K solution.
    
    Heuristic: Count lines with arithmetic operations (=, +, -, *, /)
    
    Args:
        solution: Full solution text
        
    Returns:
        Estimated number of calculation steps
    """
    # Split by sentence/line
    lines = re.split(r"[.\n]", solution)
    
    step_count = 0
    for line in lines:
        # Check for arithmetic operations
        if re.search(r"[\d+\-*/=].*[\d]", line):
            step_count += 1
    
    return max(1, step_count)  # At least 1 step


def generate_wrong_answer(correct_answer: str, method: str = "random") -> str:
    """
    Generate a plausible wrong answer.
    
    Methods:
    - "digit_swap": Swap two adjacent digits
    - "off_by_one": Add or subtract 1-10
    - "magnitude": Wrong order of magnitude
    - "random": Random choice of above
    
    Args:
        correct_answer: The correct numeric answer
        method: Generation method
        
    Returns:
        Wrong answer as string
    """
    try:
        num = float(correct_answer)
    except ValueError:
        return str(int(correct_answer) + 1) if correct_answer.isdigit() else "42"
    
    if method == "random":
        method = random.choice(["digit_swap", "off_by_one", "magnitude"])
    
    if method == "digit_swap":
        # Swap two digits
        s = correct_answer.replace(",", "")
        if len(s) >= 2:
            i = random.randint(0, len(s) - 2)
            s_list = list(s)
            s_list[i], s_list[i + 1] = s_list[i + 1], s_list[i]
            wrong = "".join(s_list)
            if wrong != correct_answer:
                return wrong
        # Fallback
        return str(int(num) + random.choice([1, -1, 10, -10]))
    
    elif method == "off_by_one":
        # Small offset
        offset = random.choice([1, 2, 5, 10, -1, -2, -5, -10])
        wrong = num + offset
        return str(int(wrong)) if wrong == int(wrong) else f"{wrong:.1f}"
    
    elif method == "magnitude":
        # Wrong order of magnitude
        factor = random.choice([0.1, 10])
        wrong = num * factor
        return str(int(wrong)) if wrong == int(wrong) else f"{wrong:.1f}"
    
    return str(int(num) + 1)


def filter_multistep_problems(
    problems: List[Dict[str, Any]],
    min_steps: int = 3,
    n_problems: int = 50,
) -> List[Dict[str, Any]]:
    """
    Filter for multi-step problems.
    
    Args:
        problems: List of problem dicts
        min_steps: Minimum calculation steps required
        n_problems: Number of problems to return
        
    Returns:
        Filtered list of problems with step counts
    """
    filtered = []
    
    for p in problems:
        n_steps = count_calculation_steps(p["answer"])
        if n_steps >= min_steps:
            filtered.append({
                **p,
                "n_steps": n_steps,
                "numeric_answer": extract_answer(p["answer"]),
            })
    
    # Sort by step count (prefer more complex) and take top n
    filtered.sort(key=lambda x: x["n_steps"], reverse=True)
    result = filtered[:n_problems]
    
    print(f"✓ Filtered to {len(result)} problems with ≥{min_steps} steps")
    return result


def prepare_problem(
    problem: Dict[str, Any],
    nonsense_reasons: Optional[List[str]] = None,
) -> GSM8KProblem:
    """
    Prepare a problem for sycophancy experiment.
    
    Args:
        problem: Raw problem dict
        nonsense_reasons: Pool of fake reasons (uses default if None)
        
    Returns:
        Structured GSM8KProblem
    """
    from .prompts import NONSENSE_REASONS
    
    if nonsense_reasons is None:
        nonsense_reasons = NONSENSE_REASONS
    
    correct = problem.get("numeric_answer") or extract_answer(problem["answer"])
    wrong = generate_wrong_answer(correct)
    reason = random.choice(nonsense_reasons)
    
    return GSM8KProblem(
        question=problem["question"],
        answer=correct,
        full_solution=problem["answer"],
        wrong_answer=wrong,
        nonsense_reason=reason,
        n_steps=problem.get("n_steps", 1),
    )


def prepare_all_problems(
    n_problems: int = 50,
    min_steps: int = 3,
    split: str = "test",
    seed: int = 42,
) -> List[GSM8KProblem]:
    """
    Load and prepare all problems for experiment.
    
    Args:
        n_problems: Number of problems to prepare
        min_steps: Minimum calculation steps
        split: Dataset split
        seed: Random seed for reproducibility
        
    Returns:
        List of prepared GSM8KProblems
    """
    random.seed(seed)
    
    # Load and filter
    raw = load_gsm8k(split)
    filtered = filter_multistep_problems(raw, min_steps, n_problems)
    
    # Prepare each
    problems = [prepare_problem(p) for p in filtered]
    
    print(f"✓ Prepared {len(problems)} problems for sycophancy experiment")
    return problems
