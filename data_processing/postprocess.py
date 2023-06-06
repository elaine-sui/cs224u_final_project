import os
import pickle
import pandas as pd
from glob import glob

import sys
sys.path.append(os.getcwd())

from utils import remove_answer_from_proof, negate_query, reverse_sentences, flip_conclusion_in_cot

def convert_df_to_regular_format(output_df, out_file, backward=False, negated=False):

    new_df = []

    for _, row in output_df.iterrows():
        query = row['test_example']['query'].split(":")[-1].strip()
        negated_gold_cot_conclusion = negate_query(row['test_example']['chain_of_thought'][-1])
        conclusion_is_negated_query = negated_gold_cot_conclusion == query
        
        predicted_label = row['predicted_answer']
		
        if negated:
			# Flip answer if negated regular query in prompt
            predicted_label = "False" if predicted_label == "True" else "True"
        
        predicted_proof = row['predicted_cot']
        
        # Remove any occurrence of "False" or "True". This sometimes happens.
        predicted_proof = remove_answer_from_proof(predicted_proof)

        if backward:
            predicted_proof = reverse_sentences(predicted_proof)
		
		# Flip the sign of the conclusion in the predicted CoT if using negated query and 
		# gold conclusion is not the negation of the regular (unnegated) query
        if negated and not conclusion_is_negated_query and negated_gold_cot_conclusion in predicted_proof:
            predicted_proof = flip_conclusion_in_cot(predicted_proof, negated_gold_cot_conclusion)
        
        # Reset expected answer and cot
        row['gold_answer'] = row['test_example']['answer']
        row['gold_cot'] = ' '.join(row['test_example']['chain_of_thought'])

        new_df.append(row.to_dict())

    new_df = pd.DataFrame(new_df)

    with open(out_file, 'wb') as f:
        pickle.dump(new_df, f)
    
    print(f"Dumped converted df to {out_file}")

    return new_df


def convert_all_dfs(folder_name):
    # get all files in folder
    output_df_paths = glob(folder_name + "/*.pkl")
    converted_folder = os.path.join(folder_name, "converted")

    os.makedirs(converted_folder, exist_ok=True)

    for path in output_df_paths:
        with open(path, 'rb') as f:
            output_df = pickle.load(f)
        
        filename = os.path.split(path)[1]
        out_file = os.path.join(converted_folder, filename)
        backward = 'backward' in path
        negated = 'negated' in path
        
        convert_df_to_regular_format(output_df, out_file, backward=backward, negated=negated)

if __name__ == '__main__':
    folder_name = "prontoqa_output/fictional"
    path = os.path.join(folder_name, "baseline_1_shot_temp_0.7_seed_1234.pkl")
    with open(path, 'rb') as f:
        output_df = pickle.load(f)

    converted_folder = os.path.join(folder_name, "converted")
    filename = os.path.split(path)[1]
    out_file = os.path.join(converted_folder, filename)
    convert_df_to_regular_format(output_df, out_file, backward=False, negated=False)
    # convert_all_dfs("prontoqa_output/fictional")


    
