import pickle
import random
import pandas as pd
from statistics import multimode

from ..utils import negate_query, reverse_sentences, flip_conclusion_in_cot

"""
Baseline (1234, 5678, 910)
Forward + backward
Forward negated + unnegated
Backward negated + unnegated
Forward randomized (1234, 5678, 910)
Backward randomized (1234, 5678, 910)

How to aggregate CoT? (write each one as a separate function -- helpful for evaluation/analysis)
 - Intersection of all CoTs (converted so that they are comparable)
 - Union of all CoTs (converted so that they are comparable)
 - Choose the longest one

How to aggregate answer? (write each one as a separate function -- helpful for evaluation/analysis)
 - Take majority answer
 - [Only applies if aggregating 2 results] 
    - If disagreement, 2 choices:
        - 1) Hard -- Must choose answer: choose answer corresponding with the longest CoT
        - 2) Soft -- Can output "I don't know"
"""

def convert_df_to_regular_format(output_df, backward=False, negated=False):

    new_df = []

    for _, row in output_df.iterrows():
        query = row['test_example']['query'].split(":")[-1].strip()
        gold_cot_conclusion = negate_query(row['test_example']['chain_of_thought'][-1])
        conclusion_is_negated_query = gold_cot_conclusion == query
        
        predicted_label = row['predicted_answer']
		
        if negated:
			# Flip answer if negated regular query in prompt
            predicted_label = "False" if predicted_label == "True" else "True"
        
        predicted_proof = row['predicted_cot']
        
        if backward:
            predicted_proof = reverse_sentences(predicted_proof)
		
		# Flip the sign of the conclusion in the predicted CoT if using negated query and 
		# gold conclusion is not the negation of the regular (unnegated) query
        if negated and not conclusion_is_negated_query:
            predicted_proof = flip_conclusion_in_cot(predicted_proof)
        
        # Reset expected answer and cot
        row['gold_answer'] = row['test_example']['answer']
        row['gold_cot'] = ' '.join(row['test_example']['chain_of_thought'])

        new_df.append(row.to_dict())

    new_df = pd.DataFrame(new_df)

    return new_df


def find_longest_cot(cots):
    # Split CoT into parts and get max length
    cot_lens = [len(cot.split('. ')) for cot in cots]

    max_len = max(cot_lens)

    # Get indices of CoTs with max len and randomly choose one if multiple
    indices = [i for i in len(cot_lens) if cot_lens[i] == max_len]
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
    # Remove the last period of the cot if exists
    cots_ = []
    for cot in cots:
        if cot[-1] == 0:
            cots_.append(cot[:-1])
        else:
            cots_.append(cot)
    
    # Separate into steps
    cot_steps = [set(cot.split('. ')) for cot in cots_]

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


def merge_cots(predicted_cots, merge_type="intersection"):
    """
    Merge chains-of-thought.
    Merge types:
        1) Intersection
        2) Union
        3) Choose longest CoT
    """
    merged_cot = None

    if merge_type in ['intersection', 'union']:
        merged_cot = cot_set_operation(predicted_cots, operation=merge_type)
    elif merge_type == 'longest':
        longest_cot_idx = find_longest_cot(predicted_cots)
        merged_cot = predicted_cots[longest_cot_idx]
    else:
        raise NotImplementedError(f'merge type {merge_type} not implemented!')
    
    return merged_cot


def merge_dfs(output_dfs, merge_answer_type, merge_cot_type, out_file):
    # Merge df predicted answers and chains-of-thought

    all_ids = output_dfs[0].id.to_list()

    merged_df = []

    for id in all_ids:
        rows = [df[df['id'] == id] for df in output_dfs]

        # assert expected answers and cots are the same
        gold_answers = [row['gold_answer'] for row in rows]
        gold_cots = [row['gold_cot'] for row in rows]

        assert len(set(gold_answers)) == 1
        assert len(set(gold_cots)) == 1

        predicted_answers = [row['predicted_answer'] for row in rows]
        predicted_cots = [row['predicted_cot'] for row in rows]

        merged_answer = merge_answers(predicted_answers, predicted_cots, merge_type=merge_answer_type)
        merged_cot = merge_cots(predicted_cots, merge_type=merge_cot_type)

        # Create new row with merged answer and cot
        row = rows[0]
        row['predicted_answer'] = merged_answer
        row['predicted_cot'] = merged_cot

        merged_df.append(row.to_dict())
    
    # Save to pkl
    with open(out_file, 'wb') as f:
        pickle.dump(merged_df, out_file)

    return merged_df





