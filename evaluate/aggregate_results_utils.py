import os
import pickle
import random
import pandas as pd
from statistics import multimode

import sys
sys.path.append(os.getcwd())
from utils import get_all_cot_steps

"""
Baseline (1234, 5678, 910)

Forward + backward
Forward negated + unnegated
Backward negated + unnegated
Forward randomized (1234, 5678, 910)
Backward randomized (1234, 5678, 910)
Forward (all types)
Backward (all types)
All (all types)

How to aggregate CoT? (write each one as a separate function -- helpful for evaluation/analysis)
 - Intersection of all CoTs (converted so that they are comparable)
 - Union of all CoTs (converted so that they are comparable)
 - Choose the longest one
 - Majority vote on the step level

How to aggregate answer? (write each one as a separate function -- helpful for evaluation/analysis)
 - Take majority answer
 - [Only applies if aggregating 2 results] 
    - If disagreement, 2 choices:
        - 1) Hard -- Must choose answer: choose answer corresponding with the longest CoT
        - 2) Soft -- Can output "I don't know"
"""

def find_longest_cot(cots):
    cot_steps = get_all_cot_steps(cots)
    # Split CoT into parts and get max length
    cot_lens = [len(cot_set) for cot_set in cot_steps]

    max_len = max(cot_lens)

    # Get indices of CoTs with max len and randomly choose one if multiple
    indices = [i for i in range(len(cot_lens)) if cot_lens[i] == max_len]
    random.shuffle(indices)
    largest_idx = indices[0]

    return largest_idx


def merge_answers(predicted_answers, predicted_cots, merge_type="hard"):
    """
    Choose majority answer.
    'hard' merge type: if disagreement, choose answer corresponding with the longest CoT
    'soft' merge type: if disagreement, write 'I don't know'
    """
    majority_answer = multimode(predicted_answers)

    if len(majority_answer) > 1: # disagreement
        if merge_type == 'hard':
            # choose answer corresponding with the longest cot
            longest_cot_idx = find_longest_cot(predicted_cots)
            majority_answer = predicted_answers[longest_cot_idx]
        elif merge_type == 'soft':
            majority_answer = "I don't know"
        else:
            raise NotImplementedError(f'merge type {merge_type} not implemented!')
    else:
        majority_answer = majority_answer[0]
    
    return majority_answer


def cot_set_operation(cots, operation="intersection"):
    # Separate into steps
    cot_steps = get_all_cot_steps(cots)

    if operation == "intersection":
        res = set.intersection(*cot_steps)
    elif operation == "union":
        res = set.union(*cot_steps)
    else:
        raise NotImplementedError(f'operation {operation} not defined')

    # FOL eval should be order invariant...
    merged_cot = list(res)
    
    # Add period at end
    merged_cot = '. '.join(merged_cot) + "."

    return merged_cot


def majority_cot_set(cots):
    # Only include steps that appear in at least 50% of the cots listed
    cot_steps = get_all_cot_steps(cots)
    all_unique_cot_steps = set.union(*cot_steps)

    majority_steps = []
    for s in all_unique_cot_steps:
        step_in_cot = [1 for cot_set in cot_steps if s in cot_set]
        if sum(step_in_cot) >= len(cots) / 2:
            majority_steps.append(s)

    merged_cot = '. '.join(majority_steps) + "."

    return merged_cot


def merge_cots(predicted_cots, merge_type="intersection"):
    """
    Merge chains-of-thought.
    Merge types:
        1) Intersection
        2) Union
        3) Choose longest CoT
        4) Majority (if the step appears in at least 50% of the CoTs)
    """
    merged_cot = None

    if merge_type in ['intersection', 'union']:
        merged_cot = cot_set_operation(predicted_cots, operation=merge_type)
    elif merge_type == 'longest':
        longest_cot_idx = find_longest_cot(predicted_cots)
        merged_cot = predicted_cots[longest_cot_idx]
    elif merge_type == 'majority':
        merged_cot = majority_cot_set(predicted_cots)
    else:
        raise NotImplementedError(f'merge type {merge_type} not implemented!')
    
    return merged_cot


def merge_dfs(output_dfs_paths, merge_answer_type, merge_cot_type, out_file):
    # Merge df predicted answers and chains-of-thought

    output_dfs = []
    for path in output_dfs_paths:
        with open(path, 'rb') as f:
            df = pickle.load(f)
            output_dfs.append(df)

    all_ids = output_dfs[0].id.to_list()

    merged_df = []

    for id in all_ids:
        rows = [df[df['id'] == id] for df in output_dfs]

        # assert expected answers and cots are the same
        gold_answers = [row['gold_answer'].item() for row in rows]
        gold_cots = [row['gold_cot'].item() for row in rows]

        assert len(set(gold_answers)) == 1
        assert len(set(gold_cots)) == 1

        predicted_answers = [row['predicted_answer'].item() for row in rows]
        
        # Remove periods from answers
        clean_predicted_answers = []
        for answer in predicted_answers:
            if answer[-1] == ".":
                answer = answer[:-1]
            
            clean_predicted_answers.append(answer)

        predicted_cots = [row['predicted_cot'].item() for row in rows]

        merged_answer = merge_answers(predicted_answers, predicted_cots, merge_type=merge_answer_type)
        merged_cot = merge_cots(predicted_cots, merge_type=merge_cot_type)

        # Create new row with merged answer and cot
        row = rows[0].iloc[0].to_dict()
        row['predicted_answer'] = merged_answer
        row['predicted_cot'] = merged_cot

        merged_df.append(row)
    
    # Save to pkl
    merged_df = pd.DataFrame(merged_df)

    with open(out_file, 'wb') as f:
        pickle.dump(merged_df, f)
    
    print(f"Dumped merged df at {out_file}")

    return merged_df