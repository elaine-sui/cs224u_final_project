import pickle
import os
from glob import glob

hops = [1, 3, 5]

def separate_output_files_by_hops(output_df_path):
    with open(output_df_path, 'rb') as f:
        output_df = pickle.load(f)

    # Separate by hops
    output_df_1hop = output_df[output_df['num_hops'] == 1]
    output_df_3hop = output_df[output_df['num_hops'] == 3]
    output_df_5hop = output_df[output_df['num_hops'] == 5]

    parent_dir, filename = os.path.split(output_df_path)
    base_filename = filename[:-4]

    # Split by aggregation type
    parent_dir = os.path.join(parent_dir, 'summary')
    os.makedirs(parent_dir, exist_ok=True)

    for hop, df in zip(hops, [output_df_1hop, output_df_3hop, output_df_5hop]):
        filename = os.path.join(parent_dir, base_filename + f"_{hop}hop.pkl")
        
        with open(filename, 'wb') as f:
            pickle.dump(df, f)

        print(f"Dumped num_hop={hop} to {filename}")

if __name__ == '__main__':
    # all the paths in the "aggregated" folder
    paths = glob("prontoqa_output/fictional/aggregated/**/*.pkl")

    for path in paths:
        print(f"Splitting file {path}")
        separate_output_files_by_hops(path)
    




