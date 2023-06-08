import networkx as nx
import pandas as pd

from run_experiment import parse_reasoning
import fol

import numpy as np
from copy import copy

from graph_utils import get_all_paths

def separate_fol_parse_into_parts(fol_parse):
    # TODO: try to make this recursive -- operand
    # Going over the FOL parse, get each part of each clause (ie, the x and y of the x ... y clauses)
    parts = []
    for fol_object in fol_parse:
        if isinstance(fol_object, fol.FOLFuncApplication):  # is a
            try:
                part = (fol_object.args[0].constant, fol_object.function)
            except:
                import pdb; pdb.set_trace()
        elif isinstance(fol_object, fol.FOLNot):  # is not
            if isinstance(fol_object.operand, fol.FOLAnd):
                part0 = fol_object.operand.operands[0].args[0].constant
                part1 = (
                    fol_object.operand.operands[0].function
                    + " "
                    + fol_object.operand.operands[1].function
                )
                part = (part0, "not " + part1)
            else:
                part = (
                    fol_object.operand.args[0].constant,
                    "not " + fol_object.operand.function,
                )
        elif isinstance(fol_object, fol.FOLForAll):  # every/each .. is (not)
            if isinstance(
                fol_object.operand.antecedent, fol.FOLAnd
            ):  # two parts to the noun
                part0 = (
                    fol_object.operand.antecedent.operands[0].function
                    + " "
                    + fol_object.operand.antecedent.operands[1].function
                )
            else:
                part0 = fol_object.operand.antecedent.function

            if isinstance(
                fol_object.operand.consequent, fol.FOLNot
            ):  # every/each .. is not
                part1 = "not " + fol_object.operand.consequent.operand.function
            elif isinstance(
                fol_object.operand.consequent, fol.FOLFuncApplication
            ):
                part1 = fol_object.operand.consequent.function
            else:
                raise NotImplementedError(
                    f"not yet encoded if {fol_object.operand.consequent} is of type {type(fol_object.operand.consequent)}"
                )

            part = (part0, part1)
        elif isinstance(fol_object, fol.FOLAnd):  # two parts to the noun
            part0 = fol_object.operands[0].args[0].constant
            part1 = (
                fol_object.operands[0].function
                + " "
                + fol_object.operands[1].function
            )
            part = (part0, part1)
        elif fol_object is None:  # not in the morphology, so skip
            continue
        else:
            print(fol_parse)
            raise NotImplementedError(
                f"not yet encoded if {fol_object} is of type {type(fol_object)}"
            )

        parts.append(part)

    return parts


def remove_intermediate_conclusion(parts, facts_and_rules_parts, query_part):
    # For instance, remove "Alex is a tumpus" if that is not the fact provided (assumed to be intermediate conclusion)
    new_parts = copy(parts)
    intermediate_conclusions = [p for p in new_parts if p[0] == query_part[0]]
    starting_statement = [
        p for p in facts_and_rules_parts if p[0] == query_part[0]
    ][0]

    while starting_statement in intermediate_conclusions:
        intermediate_conclusions.remove(starting_statement)

    for p in intermediate_conclusions:
        new_parts.remove(p)

    return new_parts


def create_reasoning_graph(merged_cot, facts_and_rules, query):
    # Parse reasoning to get FOL logic of the cot steps
    parse_errors = []
    fol_parse = parse_reasoning(merged_cot, parse_errors, keep_sentences=True)

    # Remove Nones from fol_parse
    fol_parse = [f for f in fol_parse if f[0] is not None]

    cot_steps = [p[1] for p in fol_parse]
    fol_parse = [p[0] for p in fol_parse]

    query_fol_parse = parse_reasoning(query, parse_errors)
    facts_and_rules_fol_parse = parse_reasoning(facts_and_rules, parse_errors)

    # Going over the FOL parse, get each part of each clause (ie, the x and y of the x ... y clauses)
    parts = separate_fol_parse_into_parts(fol_parse)
    # assert len(parts) == len(fol_parse)
    if len(parts) != len(fol_parse):
        import pdb

        pdb.set_trace()
        print(
            [
                (cot_steps[i], parts[i])
                for i in range(min(len(cot_steps), len(parts)))
            ]
        )
    query_part = separate_fol_parse_into_parts(query_fol_parse)[0]

    facts_and_rules_parts = separate_fol_parse_into_parts(
        facts_and_rules_fol_parse
    )

    # print(f"All parts: {parts}")

    # Create a graph using parts (edge list)
    # Remove intermediate conclusions
    all_edges = remove_intermediate_conclusion(
        parts, facts_and_rules_parts, query_part
    )
    unique_edges = set(all_edges)
    edge_weights = {edge: all_edges.count(edge) for edge in unique_edges}
    edge_data = list(unique_edges)
    for i, edge in enumerate(edge_data):
        edge_data[i] = (edge[0], edge[1], edge_weights[edge])
    weighted_edge_df = pd.DataFrame(
        columns=["source", "target", "weight"], data=edge_data
    )

    graph = nx.from_pandas_edgelist(
        weighted_edge_df, create_using=nx.DiGraph, edge_attr="weight"
    )

    return graph, parts, query_part, cot_steps


def select_path_from_reasoning_graph(
    graph, parts, query_part, cot_steps, path_selection="longest"
):
    paths = get_all_paths(graph, source=query_part[0])

    # print(f"All paths: {paths}")

    # Get all paths that start with the subject and end at the query conclusion
    path = [
        path
        for path in paths
        if (path[0][0] == query_part[0] and path[0][-1] == query_part[1])
    ]

    # If that doesn't exist, get all paths that start with the subject
    if len(path) == 0:
        path = [path for path in paths if path[0][0] == query_part[0]]

    # If that doesn't exist, get all paths
    if len(path) == 0:
        # Subject is not part of the graph. this happens if the last statement of the facts+rules is
        # not copied ino the CoT
        path = paths

    # Choose the choose the longest/shortest/highest weight path
    if len(path) > 0:
        path_lens = [len(p[0]) for p in path]
        if path_selection == "longest":  # choose longest path
            idx = np.array(path_lens).argmax()
        elif path_selection == "shortest":  # choose shortest path
            idx = np.array(path_lens).argmin()
        elif path_selection == "heaviest":  # choose the highest weighted path
            weights = [p[1] for p in path]
            idx = np.array(weights).argmax()
        else:
            raise NotImplementedError(
                f"path selection method {path_selection} not implemented yet!"
            )
        path = path[idx]
    else:  # no path -- happens if there is not intersection for example
        return ""

    path = path[0]
    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

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

    merged_cot = ". ".join(ordered_cot_steps) + "."

    return merged_cot

def run_path_selection(cot, facts_and_rules, query, path_selection):
    ## Create reasoning graph
    graph, parts, query_part, cot_steps = create_reasoning_graph(
        cot, facts_and_rules, query
    )

    ## Select appropriate path through the graph
    cot = select_path_from_reasoning_graph(
        graph, parts, query_part, cot_steps, path_selection=path_selection
    )

    return cot, graph