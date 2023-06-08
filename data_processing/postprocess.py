import os
import pickle
import pandas as pd
from glob import glob

import sys
sys.path.append(os.getcwd())

from utils import clean_answer, clean_cot_reasoning, reverse_sentences, maybe_flip_conclusion_in_cot

def convert_df_to_regular_format(output_df, out_file, backward=False, negated=False):

    new_df = []

    for _, row in output_df.iterrows():
        query = row['test_example']['query'].split(":")[-1].strip()[:-1] # remove period
        query_has_not = "not" in query
        
        predicted_label = row['predicted_answer']
        
        # Clean the predicted label
        predicted_label = clean_answer(predicted_label)
		
        if negated:
			# Flip answer if negated regular query in prompt
            predicted_label = "False" if predicted_label == "True" else "True"

        conclusion_should_have_not = (not query_has_not and predicted_label == "False") or (query_has_not and predicted_label == "True")
        
        predicted_proof = row['predicted_cot']
        # Clean the predicted CoT
        predicted_proof = clean_cot_reasoning(predicted_proof)

        if backward:
            predicted_proof = reverse_sentences(predicted_proof)
		
		# Flip the sign of the conclusion in the predicted CoT if query has a "not" in it and predicted label is True
        # or query does not have a "not" in it a predicted label is False
        predicted_proof = maybe_flip_conclusion_in_cot(predicted_proof, conclusion_should_have_not, query)
        
        # Replace predictions with converted CoT and answer
        row['predicted_cot'] = predicted_proof
        row['predicted_answer'] = predicted_label

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
    # folder_name = "prontoqa_output/fictional"
    # path = os.path.join(folder_name, "baseline_1_shot_temp_0.7_seed_1234.pkl")
    # with open(path, 'rb') as f:
    #     output_df = pickle.load(f)

    # converted_folder = os.path.join(folder_name, "converted")
    # filename = os.path.split(path)[1]
    # out_file = os.path.join(converted_folder, filename)
    # convert_df_to_regular_format(output_df, out_file, backward=False, negated=False)
    convert_all_dfs("prontoqa_output/fictional")


    
