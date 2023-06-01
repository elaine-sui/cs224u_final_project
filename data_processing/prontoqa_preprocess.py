import os
import pandas as pd
import argparse
import pickle

DATA_DIR = "prontoqa_data/fictional"
HOPS = [1, 3, 5]
JSON_PATHS = [os.path.join(dir, f"{i}hop.json") for i in HOPS]

# Load datatset
# combine all the hops into one df, add # hops as an additional column

def load_json_to_df(path, num_hops):
    df = pd.read_json(path) 
    df = df.transpose()
    df['num_hops'] = num_hops

    df = df.reset_index()
    df['id'] = df.apply(lambda x: (x['index'] + f"_{str(x['num_hops'])}"), axis=1)

    return df

def random_sample(df, k=100):
    df = df.sample(frac=1)[:k]
    return df

def preprocess(args):
    df = load_json_to_df(JSON_PATHS[0], num_hops=HOPS[0])
    df = random_sample(df, args.k)

    print(f"len(df) after adding {JSON_PATHS[0]}: {len(df)}")
    for hops, path in zip(HOPS[1:], JSON_PATHS[1:]):
        df2 = load_json_to_df(path, num_hops=hops) # starting from the 2nd
        df2 = random_sample(df2, args.k)
        df = pd.concat([df, df2])
        print(f"len(df) after adding {path}: {len(df)}")

    df = df.reset_index(drop=True)

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=100, type=int, help='number of examples to sample per hop')
    args = parser.parse_args()

    df = preprocess(args)

    out_file = os.path.join(DATA_DIR, 'sampled_data.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(df, f)
