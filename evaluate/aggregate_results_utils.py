import os
import pickle
import random
import pandas as pd
import re
from statistics import multimode

from path_selection import run_path_selection

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

    if len(majority_answer) > 1:  # disagreement
        if merge_type == "hard":
            # choose answer corresponding with the longest cot
            longest_cot_idx = find_longest_cot(predicted_cots)
            majority_answer = predicted_answers[longest_cot_idx]
        elif merge_type == "soft":
            majority_answer = "I don't know"
        else:
            raise NotImplementedError(
                f"merge type {merge_type} not implemented!"
            )
    else:
        majority_answer = majority_answer[0]

    return majority_answer


def concatenate_cots(cots):
    cot_steps = get_all_cot_steps(cots, save_duplicates=True)
    merged_cot = []
    for cot_lst in cot_steps:
        merged_cot.extend(cot_lst)

    return merged_cot


def cot_set_operation(cots, operation="intersection"):
    # Separate into steps
    cot_steps = get_all_cot_steps(cots)

    if operation == "intersection":
        res = set.intersection(*cot_steps)
    elif operation == "union":
        res = set.union(*cot_steps)
    else:
        raise NotImplementedError(f"operation {operation} not defined")

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


def merge_cots(
    predicted_cots,
    facts_and_rules,
    query,
    merge_type="intersection",
    path_selection="longest",
):
    """
    Merge chains-of-thought.
    Merge types (ie preprocessing the CoT steps before reasoning graph creation):
        1) Intersection
        2) Union
        3) Choose longest CoT
        4) Majority (if the step appears in at least 50% of the CoTs)
        5) None
    Path selection types:
        1) Longest
        2) Shortest
        3) Heaviest
    """
    merged_cot = None

    if (
        merge_type == "none"
    ):  # just concatenate all the CoTs together because want duplicates for weights for path selection
        merged_cot = concatenate_cots(predicted_cots)
    elif merge_type in ["intersection", "union"]:
        merged_cot = cot_set_operation(predicted_cots, operation=merge_type)
    elif merge_type == "longest":
        longest_cot_idx = find_longest_cot(predicted_cots)
        merged_cot = predicted_cots[longest_cot_idx]
    elif merge_type == "majority":
        merged_cot = majority_cot_set(predicted_cots)
    else:
        raise NotImplementedError(f"merge type {merge_type} not implemented!")

    if isinstance(merged_cot, list):
        merged_cot = ". ".join(merged_cot) + "."

    print(f"After merging (before ordering): {merged_cot}")

    # Re-order merged CoT steps
    graph = None
    if path_selection != "none":
        # Run path selection
        merged_cot, graph = run_path_selection(merged_cot, facts_and_rules, query, path_selection=path_selection)

    return merged_cot, graph


def merge_dfs(
    output_dfs_paths,
    merge_answer_type,
    merge_cot_of_majority_answer,
    merge_cot_type,
    path_selection,
    out_file,
):
    # Merge df predicted answers and chains-of-thought

    output_dfs = []
    for path in output_dfs_paths:
        with open(path, "rb") as f:
            df = pickle.load(f)
            output_dfs.append(df)

    all_ids = output_dfs[0].id.to_list()

    merged_df = []

    for id in all_ids:
        rows = [df[df["id"] == id] for df in output_dfs]

        # assert expected answers and cots are the same
        gold_answers = [row["gold_answer"].item() for row in rows]
        gold_cots = [row["gold_cot"].item() for row in rows]

        assert len(set(gold_answers)) == 1
        assert len(set(gold_cots)) == 1

        predicted_answers = [row["predicted_answer"].item() for row in rows]
        predicted_cots = [row["predicted_cot"].item() for row in rows]

        query = rows[0]["test_example"].item()["query"].split(":")[-1].strip()

        facts_and_rules = rows[0]["test_example"].item()["question"]

        print(f"Query: {query}")

        merged_answer = merge_answers(
            predicted_answers, predicted_cots, merge_type=merge_answer_type
        )

        if merge_cot_of_majority_answer:
            majority_answer_indices = [i for i in range(len(predicted_answers)) if predicted_answers[i] == merged_answer]
            predicted_cots = [predicted_cots[i] for i in majority_answer_indices]

        print(f"Predicted CoTs: {predicted_cots}")

        merged_cot, graph = merge_cots(
            predicted_cots,
            facts_and_rules,
            query,
            merge_type=merge_cot_type,
            path_selection=path_selection,
        )

        print(f"Merged CoT: {merged_cot}")
        print(f"Gold CoT: {gold_cots[0]}")

        # Create new row with merged answer and cot
        row = rows[0].iloc[0].to_dict()
        row["predicted_answer"] = merged_answer
        row["predicted_cot"] = merged_cot
        row["all_predicted_cots"] = predicted_cots
        row["all_predicted_answers"] = predicted_answers
        row["graph"] = graph

        merged_df.append(row)
        print("=" * 80 + "\n")

    # Save to pkl
    merged_df = pd.DataFrame(merged_df)

    with open(out_file, "wb") as f:
        pickle.dump(merged_df, f)

    print(f"Dumped merged df at {out_file}")

    return merged_df

def merge_unnegated_negated_dfs(
    output_dfs_paths,
    path_selection,
    out_file,
):
    # Merge df predicted answers and chains-of-thought

    output_dfs = {}
    for path in output_dfs_paths:
        with open(path, "rb") as f:
            df = pickle.load(f)
        
        if "_negated_" in path:
            output_dfs["negated"] = df
        else:
            output_dfs["unnegated"] = df

    all_ids = output_dfs["negated"].id.to_list()

    merged_df = []

    for id in all_ids:
        rows = {k: df[df["id"] == id] for k, df in output_dfs.items()}

        # assert expected answers and cots are the same
        gold_answers = [row["gold_answer"].item() for row in rows.values()]
        gold_cots = [row["gold_cot"].item() for row in rows.values()]

        assert len(set(gold_answers)) == 1
        assert len(set(gold_cots)) == 1

        predicted_answers = {k: row["predicted_answer"].item() for k, row in rows.items()}
        predicted_cots = {k: row["predicted_cot"].item() for k, row in rows.items()}

        query = rows["negated"]["test_example"].item()["query"].split(":")[-1].strip()

        facts_and_rules = rows["negated"]["test_example"].item()["question"]

        print(f"Query: {query}")

        if "not" in query:
            merged_answer = predicted_answers["negated"] # double negation cancels out
            merged_cot = predicted_cots["negated"]
        else:
            merged_answer = predicted_answers["unnegated"]
            merged_cot = predicted_cots["unnegated"]

        print(f"Predicted CoTs: {predicted_cots}")

        # Path selection
        graph = None
        if path_selection != "none":
            # Run path selection
            merged_cot, graph = run_path_selection(merged_cot, facts_and_rules, query, path_selection=path_selection)

        print(f"Merged CoT: {merged_cot}")
        print(f"Gold CoT: {gold_cots[0]}")

        # Create new row with merged answer and cot
        row = rows["negated"].iloc[0].to_dict()
        row["predicted_answer"] = merged_answer
        row["predicted_cot"] = merged_cot
        row["all_predicted_cots"] = predicted_cots
        row["all_predicted_answers"] = predicted_answers
        row["graph"] = graph

        merged_df.append(row)
        print("=" * 80 + "\n")

    # Save to pkl
    merged_df = pd.DataFrame(merged_df)

    with open(out_file, "wb") as f:
        pickle.dump(merged_df, f)

    print(f"Dumped merged df at {out_file}")

    return merged_df
