import os
import argparse

from aggregate_results_utils import merge_dfs, merge_unnegated_negated_dfs

ROOT = '/sailhome/esui/cs224u_final_project/prontoqa_output/fictional'

OUT_FOLDER = os.path.join(ROOT, 'aggregated')
os.makedirs(OUT_FOLDER, exist_ok=True)

FILES = {
        'baseline_seed1234': 'baseline_1_shot_temp_0.0_seed_1234.pkl',
        'baseline_seed5678': 'baseline_1_shot_temp_0.7_seed_5678.pkl',
        'baseline_seed910': 'baseline_1_shot_temp_0.7_seed_910.pkl',
        'baseline_neg': 'baseline_negated_1_shot_temp_0.0_seed_1234.pkl',
        'forward_ltsbs': 'forward_ltsbs_1_shot_temp_0.0_seed_1234.pkl',
        'forward_neg_ltsbs': 'forward_negated_ltsbs_1_shot_temp_0.0_seed_1234.pkl',
        'forward_1': 'forward_ltsbs_randomized_order_1_shot_temp_0.0_seed_1234.pkl',
        'forward_2': 'forward_ltsbs_randomized_order_1_shot_temp_0.0_seed_12345.pkl',
        'backward_ltsbs': 'backward_ltsbs_1_shot_temp_0.0_seed_1234.pkl',
        'backward_neg_ltsbs': 'backward_negated_ltsbs_1_shot_temp_0.0_seed_1234.pkl',
        'backward_1': 'backward_ltsbs_randomized_order_1_shot_temp_0.0_seed_1234.pkl',
        'backward_2': 'backward_ltsbs_randomized_order_1_shot_temp_0.0_seed_12345.pkl',
    }

AGGREGATION_TYPES = [
    "single_baseline",
    "single_forward",
    "single_backward",
    "single_baseline_neg",
    "single_forward_neg",
    "single_backward_neg",
    "double_baseline_negation",
    "double_forward_negation",
    "double_backward_negation",
    "baseline",
    "direction", 
    "forward_negation", 
    "backward_negation", 
    "forward_randomized_order", 
    "backward_randomized_order", 
    "forward_all", 
    "backward_all",
    "all",
]

MERGE_ANSWER_TYPES = ['hard', 'soft']
MERGE_COT_TYPES = ['intersection', 'union', 'longest', 'majority', 'none']
PATH_SELECTION_TYPES = ['longest', 'shortest', 'heaviest', 'none']

def get_df_paths_and_out_file(aggregation_type, merge_answer_type, merge_cot_of_majority_answer, merge_cot_type, path_selection):
    out_folder = os.path.join(OUT_FOLDER, aggregation_type)
    if merge_cot_of_majority_answer:
        out_folder = os.path.join(out_folder, "merge_cot_of_majority_answer")
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, f'merge_answer_{merge_answer_type}_merge_cot_{merge_cot_type}_path_select_{path_selection}.pkl')

    if aggregation_type == "single_baseline":
        df_paths = [
            FILES['baseline_seed1234']
        ]
    elif aggregation_type == "single_baseline_neg":
        df_paths = [
            FILES['baseline_neg']
        ]
    elif aggregation_type == "single_forward":
        df_paths = [
            FILES['forward_ltsbs']
        ]
    elif aggregation_type == "single_backward":
        df_paths = [
            FILES['backward_ltsbs']
        ]
    elif aggregation_type == "single_forward_neg":
        df_paths = [
            FILES['forward_neg_ltsbs']
        ]
    elif aggregation_type == "single_backward_neg":
        df_paths = [
            FILES['backward_neg_ltsbs']
        ]
    elif aggregation_type == "double_baseline_negation":
        df_paths = [
            FILES['baseline_seed1234'],
            FILES['baseline_neg']
        ]
    elif aggregation_type == "double_forward_negation":
        df_paths = [
            FILES['forward_ltsbs'],
            FILES['forward_neg_ltsbs']
        ]
    elif aggregation_type == "double_backward_negation":
        df_paths = [
            FILES['backward_ltsbs'],
            FILES['backward_neg_ltsbs']
        ]
    elif aggregation_type == "baseline":
        df_paths = [ 
            FILES['baseline_seed1234'],
            FILES['baseline_seed5678'],
            FILES['baseline_seed910']
        ]
    elif aggregation_type == "direction":
        df_paths = [
            FILES['baseline_seed1234'],
            FILES['forward_ltsbs'], 
            FILES['backward_ltsbs']
        ]
    elif aggregation_type == "forward_negation":
        df_paths = [
            FILES['baseline_seed1234'],
            FILES['forward_ltsbs'],
            FILES['forward_neg_ltsbs']
        ]
    elif aggregation_type == "backward_negation":
        df_paths = [
            FILES['baseline_seed1234'],
            FILES['backward_ltsbs'],
            FILES['backward_neg_ltsbs']
        ]
    elif aggregation_type == "forward_randomized_order":
        df_paths = [
            FILES['forward_ltsbs'],
            FILES['forward_1'],
            FILES['forward_2'],
        ]
    elif aggregation_type == "backward_randomized_order":
        df_paths = [
            FILES['backward_ltsbs'],
            FILES['backward_1'],
            FILES['backward_2'],
        ]
    elif aggregation_type == "forward_all":
        df_paths = [
            FILES['forward_ltsbs'],
            FILES['forward_1'],
            FILES['forward_2'],
            FILES['forward_neg_ltsbs'],
            FILES['forward_neg_ltsbs'],
        ]
    elif aggregation_type == "backward_all":
        df_paths = [
            FILES['backward_ltsbs'],
            FILES['backward_1'],
            FILES['backward_2'],
            FILES['backward_neg_ltsbs'],
            FILES['backward_neg_ltsbs'],
        ]
    elif aggregation_type == "all":
        df_paths = [
            FILES['baseline_seed1234'],
            FILES['baseline_seed5678'],
            FILES['baseline_neg'],
            FILES['forward_ltsbs'],
            FILES['forward_neg_ltsbs'],
            FILES['backward_ltsbs'],
            FILES['backward_neg_ltsbs'],
        ]
    
    df_paths = [os.path.join(ROOT, 'converted', path) for path in df_paths]

    return df_paths, out_file

def aggregate(aggregation_type, merge_answer_type, merge_cot_of_majority_answer, merge_cot_type, path_selection):
    df_paths, out_file = get_df_paths_and_out_file(aggregation_type, merge_answer_type, merge_cot_of_majority_answer, merge_cot_type, path_selection)
    
    if "double" in aggregation_type:
        merge_unnegated_negated_dfs(
            output_dfs_paths=df_paths,
            path_selection=path_selection,
            out_file=out_file
        )
    else:
        merge_dfs(
            output_dfs_paths=df_paths,
            merge_answer_type=merge_answer_type,
            merge_cot_of_majority_answer=merge_cot_of_majority_answer,
            merge_cot_type=merge_cot_type,
            path_selection=path_selection,
            out_file=out_file,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation_type", type=str, choices=AGGREGATION_TYPES)
    parser.add_argument("--merge_answer_type", type=str, choices=MERGE_ANSWER_TYPES)
    parser.add_argument("--merge_cot_of_majority_answer", action="store_true")
    parser.add_argument("--merge_cot_type", type=str, choices=MERGE_COT_TYPES)
    parser.add_argument("--path_selection", type=str, choices=PATH_SELECTION_TYPES)

    args = parser.parse_args()

    if args.path_selection == 'heaviest' and args.merge_cot_type != 'none':
        print("Path selection type heaviest requires merge cot type to be none")
        exit(1)

    aggregate(
        aggregation_type=args.aggregation_type, 
        merge_answer_type=args.merge_answer_type, 
        merge_cot_of_majority_answer=args.merge_cot_of_majority_answer,
        merge_cot_type=args.merge_cot_type, 
        path_selection=args.path_selection,
    )
