import os
import argparse

from aggregate_results_utils import merge_dfs

ROOT = '/sailhome/esui/cs224u_final_project/prontoqa_output/fictional'

OUT_FOLDER = os.path.join(ROOT, 'aggregated')
os.makedirs(OUT_FOLDER, exist_ok=True)

FILES = {
        'forward_0': 'forward_1_shot_temp_0.0_seed_1234.pkl',
        'forward_1': 'forward_randomized_order_1_shot_temp_0.0_seed_1234.pkl',
        'forward_2': 'forward_randomized_order_1_shot_temp_0.0_seed_12345.pkl',
        'backward_0': 'backward_1_shot_temp_0.0_seed_1234.pkl',
        'backward_1': 'backward_randomized_order_1_shot_temp_0.0_seed_1234.pkl',
        'backward_2': 'backward_randomized_order_1_shot_temp_0.0_seed_12345.pkl',
        'forward_neg': 'forward_negated_1_shot_temp_0.0_seed_1234.pkl',
        'backward_neg': 'backward_negated_1_shot_temp_0.0_seed_1234.pkl',
        'baseline_seed1234': 'baseline_1_shot_temp_0.7_seed_1234.pkl',
        'baseline_seed5678': 'baseline_1_shot_temp_0.7_seed_5678.pkl',
        'baseline_seed910': 'baseline_1_shot_temp_0.7_seed_910.pkl',
    }

AGGREGATION_TYPES = [
    "direction", 
    "forward_negation", 
    "backward_negation", 
    "forward_randomized_order", 
    "backward_randomized_order", 
    "forward_all", 
    "backward_all",
    "all"
]

MERGE_ANSWER_TYPES = ['hard', 'soft']
MERGE_COT_TYPES = ['intersection', 'union', 'longest', 'majority', 'none']
PATH_SELECTION_TYPES = ['longest', 'shortest', 'heaviest']

def get_df_paths_and_out_file(aggregation_type, merge_answer_type, merge_cot_type, path_selection):
    out_file = os.path.join(OUT_FOLDER, f'{aggregation_type}_consistency_merge_answer_{merge_answer_type}_merge_cot_{merge_cot_type}_path_select_{path_selection}.pkl')

    if aggregation_type == "direction":
        df_paths = [
            FILES['forward_0'], 
            FILES['backward_0']
        ]
    elif aggregation_type == "forward_negation":
        df_paths = [
            FILES['forward_0'],
            FILES['forward_neg']
        ]
    elif aggregation_type == "backward_negation":
        df_paths = [
            FILES['backward_0'],
            FILES['backward_neg']
        ]
    elif aggregation_type == "forward_randomized_order":
        df_paths = [
            FILES['forward_0'],
            FILES['forward_1'],
            FILES['forward_2'],
        ]
    elif aggregation_type == "backward_randomized_order":
        df_paths = [
            FILES['backward_0'],
            FILES['backward_1'],
            FILES['backward_2'],
        ]
    elif aggregation_type == "forward_all":
        df_paths = [
            FILES['forward_0'],
            FILES['forward_1'],
            FILES['forward_2'],
            FILES['forward_neg'],
        ]
    elif aggregation_type == "backward_all":
        df_paths = [
            FILES['backward_0'],
            FILES['backward_1'],
            FILES['backward_2'],
            FILES['backward_neg'],
        ]
    elif aggregation_type == "all":
        df_paths = list(FILES.values())
    
    df_paths = [os.path.join(ROOT, 'converted', path) for path in df_paths]

    return df_paths, out_file

def aggregate(aggregation_type, merge_answer_type, merge_cot_type, path_selection):
    df_paths, out_file = get_df_paths_and_out_file(aggregation_type, merge_answer_type, merge_cot_type, path_selection)

    merge_dfs(df_paths, merge_answer_type, merge_cot_type, path_selection, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation_type", type=str, choices=AGGREGATION_TYPES)
    parser.add_argument("--merge_answer_type", type=str, choices=MERGE_ANSWER_TYPES)
    parser.add_argument("--merge_cot_type", type=str, choices=MERGE_COT_TYPES)
    parser.add_argument("--path_selection", type=str, choices=PATH_SELECTION_TYPES)

    args = parser.parse_args()

    if args.path_selection == 'heaviest' and args.merge_cot_type != 'none':
        print("Path selection type heaviest requires merge cot type to be none")
        exit(1)

    aggregate(args.aggregation_type, args.merge_answer_type, args.merge_cot_type, args.path_selection)
