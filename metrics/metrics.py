from typing import List
import numpy as np

def strict_acc(preds: List[str], labels: List[str]):
    # Strict accuracy
    num_correct = sum(np.array(preds) == np.array(labels))
    acc = num_correct / len(preds)
    return acc

def hallucination_count(pred_cot: List[str], premise: List[str], query: str):
    # number of steps that are hallucinated
    given_steps = set(premise + [query]) 
    hallucinated_steps = set(pred_cot) - set(given_steps)
    return len(hallucinated_steps)

def skipped_steps_count(pred_cot, gold_cot):
    # number of steps that are skipped over
    return len(set(gold_cot) - set(pred_cot))

def cot_precision(pred_cot, gold_cot):
    correct_steps = set(pred_cot).intersection(set(gold_cot))
    return len(correct_steps) / len(pred_cot)

def cot_recall(pred_cot, gold_cot):
    correct_steps = set(pred_cot).intersection(set(gold_cot))
    return len(correct_steps) / len(gold_cot)
        