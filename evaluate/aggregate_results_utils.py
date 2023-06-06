import os
import pickle
import random
import pandas as pd
import numpy as np
from copy import copy
import re

from statistics import multimode
import networkx as nx

from run_experiment import parse_reasoning
import fol

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
    try:
        cot_steps = get_all_cot_steps(cots)
    except:
        import pdb; pdb.set_trace()

    if operation == "intersection":
        res = set.intersection(*cot_steps)
    elif operation == "union":
        res = set.union(*cot_steps)
    else:
        raise NotImplementedError(f'operation {operation} not defined')

    merged_cot = list(res)

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

    return majority_steps


def separate_fol_parse_into_parts(fol_parse):
    # TODO: make this recursive -- operand
    # Going over the FOL parse, get each part of each clause (ie, the x and y of the x ... y clauses)
    parts = []
    for fol_object in fol_parse:
        if isinstance(fol_object, fol.FOLFuncApplication): # is a
            try:
                part = (fol_object.args[0].constant, fol_object.function)
            except:
                import pdb; pdb.set_trace()
        elif isinstance(fol_object, fol.FOLNot): # is not
            if isinstance(fol_object.operand, fol.FOLAnd):
                part0 = fol_object.operand.operands[0].args[0].constant
                part1 = fol_object.operand.operands[0].function + " " + fol_object.operand.operands[1].function
                part = (part0, "not " + part1)
            else:
                part = (fol_object.operand.args[0].constant, "not " + fol_object.operand.function)
        elif isinstance(fol_object, fol.FOLForAll): # every/each .. is (not)
            if isinstance(fol_object.operand.antecedent, fol.FOLAnd): # two parts to the noun
                part0 = fol_object.operand.antecedent.operands[0].function + " " + fol_object.operand.antecedent.operands[1].function
            else:
                part0 = fol_object.operand.antecedent.function
            
            if isinstance(fol_object.operand.consequent, fol.FOLNot): # every/each .. is not
                part1 = "not " + fol_object.operand.consequent.operand.function
            elif isinstance(fol_object.operand.consequent, fol.FOLFuncApplication):
                part1 = fol_object.operand.consequent.function
            else:
                raise NotImplementedError(f"not yet encoded if {fol_object.operand.consequent} is of type {type(fol_object.operand.consequent)}")
            
            part = (part0, part1)
        elif isinstance(fol_object, fol.FOLAnd): # two parts to the noun
            part0 = fol_object.operands[0].args[0].constant
            part1 = fol_object.operands[0].function + " " + fol_object.operands[1].function
            part = (part0, part1)
        elif fol_object is None: # not in the morphology, so skip
            continue
        else:
            print(fol_parse)
            raise NotImplementedError(f"not yet encoded if {fol_object} is of type {type(fol_object)}")
        
        parts.append(part)
    
    return parts


def get_all_paths(G, source):
    paths = []

    if source in G.nodes:
        for node in G.nodes:
            if node == source:
                continue
            for path in nx.all_simple_paths(G, source, node):
                paths.append(path)
    else:
        for node in G.nodes :
            for node2 in G.nodes:
                if node2 == node:
                    continue
                for path in nx.all_simple_paths(G, node, node2) :
                    paths.append(path)
    return paths


def remove_intermediate_conclusion(parts, facts_and_rules_parts, query_part):
    # For instance, remove "Alex is a tumpus" if that is not the fact provided (assumed to be intermediate conclusion)
    new_parts = copy(parts)
    intermediate_conclusions = [p for p in new_parts if p[0] == query_part[0]]
    starting_statement = [p for p in facts_and_rules_parts if p[0] == query_part[0]][0]

    if starting_statement in intermediate_conclusions:
        intermediate_conclusions.remove(starting_statement)

    for p in intermediate_conclusions:
        new_parts.remove(p)
    
    return new_parts


def order_cot_from_steps(merged_cot, facts_and_rules, query):
    cot_steps = get_all_cot_steps([merged_cot], list=True)[0]
    # Parse reasoning to get FOL logic of the cot steps
    parse_errors = []
    fol_parse = parse_reasoning(merged_cot, parse_errors)
    query_fol_parse = parse_reasoning(query, parse_errors)
    facts_and_rules_fol_parse = parse_reasoning(facts_and_rules, parse_errors)
    
    # Going over the FOL parse, get each part of each clause (ie, the x and y of the x ... y clauses)
    parts = separate_fol_parse_into_parts(fol_parse)
    query_part = separate_fol_parse_into_parts(query_fol_parse)[0]

    facts_and_rules_parts = separate_fol_parse_into_parts(facts_and_rules_fol_parse)

    # print(f"All parts: {parts}")

    # Create a graph using parts (edge list)
    # Remove intermediate conclusions
    parts_cpy = remove_intermediate_conclusion(parts, facts_and_rules_parts, query_part)

    graph = nx.from_edgelist(parts_cpy, create_using=nx.DiGraph)
    paths = get_all_paths(graph, source=query_part[0])

    # print(f"All paths: {paths}")

    # Choose the path that ends with the edge query_part
    path = [path for path in paths if (path[0] == query_part[0] and path[-1] == query_part[1])]

    # If that doesn't exist, choose the longest path that starts with the subject
    if len(path) == 0:
        path = [path for path in paths if path[0] == query_part[0]]
    
    # If that doesn't exist, choose the longest path
    if len(path) == 0:
        # Subject is not part of the graph. this happens if the last statement of the facts+rules is
        # not copied ino the CoT
        path = paths
    
    if len(path) > 0:
        # choose longest path
        path_lens = [len(p) for p in path]
        max_idx = np.array(path_lens).argmax()
        path = path[max_idx]
    else: # no path -- happens if there is not intersection for example
        return ""

    edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]

    # Same format as gold CoT where every implication is followed by a conclusion
    # E.g. "Max is a tumpus. Every tumpus is a jompus. Max is a jompus. Jompuses are vumpuses. Max is a vumpus."

    all_edges = [edges[0]]
    for edge in edges[1:]:
        all_edges.append(edge)
        all_edges.append((edges[0][0], edge[1]))
    
    # print(f"All edges: {all_edges}")

    new_order = []
    for e in all_edges:
        if e in parts:
            new_order.append(parts.index(e))
    ordered_cot_steps = [cot_steps[i] for i in new_order]

    merged_cot = '. '.join(ordered_cot_steps) + "."

    return merged_cot


def merge_cots(predicted_cots, facts_and_rules, query, merge_type="intersection"):
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

    if isinstance(merged_cot, list):
        merged_cot = '. '.join(merged_cot) + "."
    
    print(f"After merging (before ordering): {merged_cot}")

    # Re-order merged CoT steps
    merged_cot = order_cot_from_steps(merged_cot, facts_and_rules, query)

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
        
        clean_predicted_answers = predicted_answers

        predicted_cots = [row['predicted_cot'].item() for row in rows]

        # Remove all occurrences of "Therefore, " and double negatives from predicted_cots
        clean_predicted_cots = []
        for cot in predicted_cots:
            clean_cot = re.sub("Therefore, ", "", cot)
            clean_cot = re.sub("However, ", "", clean_cot)
            clean_cot = re.sub("not not ", "", clean_cot)
            clean_cot = re.sub("Since ", "", clean_cot)
            clean_cot = re.sub(" and ", ",", clean_cot)
            clean_cot = clean_cot.split(",")

            if isinstance(clean_cot, list):
                for c in clean_cot:
                    clean_predicted_cots.append(c.strip().capitalize())
            else:
                clean_predicted_cots.append(clean_cot.strip().capitalize())
        
        predicted_cots = clean_predicted_cots

        query = rows[0]['test_example'].item()['query'].split(':')[-1].strip()

        facts_and_rules = rows[0]['test_example'].item()['question']

        print(f"Query: {query}")
        # print(f"Predicted CoTs: {predicted_cots}")

        merged_answer = merge_answers(predicted_answers, predicted_cots, merge_type=merge_answer_type)
        merged_cot = merge_cots(predicted_cots, facts_and_rules, query, merge_type=merge_cot_type)

        print(f"Merged CoT: {merged_cot}")
        print(f"Gold CoT: {gold_cots[0]}")

        # Create new row with merged answer and cot
        row = rows[0].iloc[0].to_dict()
        row['predicted_answer'] = merged_answer
        row['predicted_cot'] = merged_cot

        merged_df.append(row)
        print("="*80 + "\n")
    
    # Save to pkl
    merged_df = pd.DataFrame(merged_df)

    with open(out_file, 'wb') as f:
        pickle.dump(merged_df, f)
    
    print(f"Dumped merged df at {out_file}")

    return merged_df