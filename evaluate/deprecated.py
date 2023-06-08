
def order_fol_object_parts(parts, query_part):
    # TODO
    # Order the parts such that 2nd part is the last tuple is the 1st part of the next tuple (e.g. [(a, b), (b, c), (c, d)])

    df = pd.DataFrame(columns=['index0', 'index1'], data=parts)

    # Find the start (tuple such that value in index0 appears twice and value in index1 also appears in index0 for another tuple)
    index0_value_counts = df['index0'].value_counts()
    index0_twice = [i for i in index0_value_counts.index if index0_value_counts[i] == 2] # values that appear twice
    possible_starting_positions = [i for i in df.index if df.iloc[i].index0 in index0_twice and df.iloc[i].index1 in df.index0.to_list()]
    possible_ending_positions = [i for i in df.index if df.iloc[i].index0 == query_part[0] and df.iloc[i].index1 == query_part[1]]

    if len(possible_starting_positions) == 0: # original heuristic didn't work. so take start/end pos based off of query
        possible_starting_positions = [i for i in df.index if df.iloc[i].index0 not in df.index1.to_list() and df.iloc[i].index0 == query_part[0]]

    if len(possible_ending_positions) == 0: # didn't reach conclusion. relax constraint
        possible_ending_positions = [i for i in df.index if df.iloc[i].index1 not in df.index0.to_list() and df.iloc[i].index1 == query_part[1]]
    
    if len(possible_ending_positions) == 0: # relax constraint even more
        possible_ending_positions = [i for i in df.index if df.iloc[i].index1 not in df.index0.to_list()]

    if len(possible_starting_positions) != 1 or len(possible_ending_positions) != 1:
        ### TODO
        # not sure what to do here...
        import pdb; pdb.set_trace()
        print("Possible start/end pos not 1")

    start_pos = possible_starting_positions[0]
    end_pos = possible_ending_positions[0]

    ordering = [[]]
    while df.iloc[start_pos].index1 != df.iloc[end_pos].index1:
        ordering.append(start_pos)
        start_pos = [i for i in df.index if df.iloc[start_pos].index1 == df.iloc[i].index0]

        if len(start_pos) > 1:
            ### TODO
            # not sure what to do here...

            if end_pos in start_pos:
                start_pos = end_pos
            else:
                print("Too many possible start pos. Keep all options open?")
                import pdb; pdb.set_trace()
        elif len(start_pos) == 0: # steps skipped
            break
        else:
            start_pos = start_pos[0]
    
    if start_pos != []:
        ordering.extend([start_pos, end_pos])
    else:
        ordering.append(end_pos)

    return ordering


ROOT = '../prontoqa_output/fictional/'

RANDOMIZED_ORDER_FILES = {
        'forward_0': 'forward_1_shot_temp_0.0_seed_1234.pkl',
        'forward_1': 'forward_randomized_order_1_shot_temp_0.0_seed_1234.pkl',
        'forward_2': 'forward_randomized_order_1_shot_temp_0.0_seed_12345.pkl',
        'backward_0': 'backward_1_shot_temp_0.0_seed_1234.pkl',
        'backward_1': 'backward_randomized_order_1_shot_temp_0.0_seed_1234.pkl',
        'backward_2': 'backward_randomized_order_1_shot_temp_0.0_seed_12345.pkl'
    }

NEGATED_FILES = {
        'forward_pos': 'forward_1_shot_temp_0.0_seed_1234.pkl',
        'forward_neg': 'forward_negated_1_shot_temp_0.0_seed_1234.pkl',
        'backward_pos': 'backward_1_shot_temp_0.0_seed_1234.pkl',
        'backward_neg': 'backward_negated_1_shot_temp_0.0_seed_1234.pkl'
    }

BASELINE = {
        'baseline_seed1234': 'baseline_1_shot_temp_0.7_seed_1234.pkl',
        'baseline_seed5678': 'baseline_1_shot_temp_0.7_seed_5678.pkl',
        'baseline_seed910': 'baseline_1_shot_temp_0.7_seed_910.pkl',
    }
        
def answer_majority_unknown(df):
    mode = df.mode(axis = 1)
    na_idx = mode[1].isna()
    mode.loc[~na_idx] = np.nan
    return mode.iloc[:, 0] # TODO: currently outputs "I don't know" a.k.a NaN!

answer_majority = answer_majority_unknown

def evaluate_cot_prop_exact_matches(pred_cot, gold_cot):
    # Extremely simple CoT checker, checks proportion of facts in prediction that
    # are in gold.
    pred_cot = pred_cot.split('. ')
    gold_cot = gold_cot.split('. ')
    return pd.Series(pred_cot).isin(gold_cot).mean()

def choose_cot_intersection(*cots):
    pass # TODO

def choose_cot_union(*cots):
    return '---'.join(cots)

def choose_cot_longest(*cots):
    return max(cots, key = len)

choose_cot = choose_cot_longest

def evaluate_randomized_order_answer():
    """
    Returns a tuple with:
    (accuracy of all random orders where zero'th is original order, concatenated DataFrame with CoT etc.)
    """
    dfs = {}
    for k, v in RANDOMIZED_ORDER_FILES.items():
        dfs[k] = pickle.load(open(ROOT + v, 'rb'))
        dfs[k]['predicted_answer'] = dfs[k]['predicted_answer'].replace({'True': True, 'False': False})
        dfs[k]['gold_answer'] = dfs[k]['gold_answer'].replace({'True': True, 'False': False})

    forward_concat = pd.concat([df['predicted_answer'] for k, df in dfs.items() if 'forward' in k], axis = 1)
    backward_concat = pd.concat([df['predicted_answer'] for k, df in dfs.items() if 'backward' in k], axis = 1)

    forward_maj = answer_majority(forward_concat)
    backward_maj = answer_majority(backward_concat)

    dfs['forward_maj'] = dfs['forward_0'].copy()
    dfs['backward_maj'] = dfs['backward_0'].copy()

    dfs['forward_maj']['predicted_answer'] = forward_maj
    dfs['backward_maj']['predicted_answer'] = backward_maj

    for k, df in dfs.items():
        df['key'] = k
        df['correct'] = (df['predicted_answer'] == df['gold_answer'])

    df_concat = pd.concat([df for df in dfs.values()], axis = 0)
    return pd.pivot_table(df_concat, index = 'key', columns = 'num_hops', values = 'correct', margins = True) * 100, df_concat

def evaluate_negated_answer():
    """
    Returns a tuple with:
    (accuracy of forward/backward w/ and w/o negation, concatenated DataFrame with CoT etc.)
    """
    dfs = {}
    for k, v in NEGATED_FILES.items():
        dfs[k] = pickle.load(open(ROOT + v, 'rb'))
        dfs[k]['predicted_answer'] = dfs[k]['predicted_answer'].replace({'True': True, 'False': False})
        dfs[k]['gold_answer'] = dfs[k]['gold_answer'].replace({'True': True, 'False': False})

    forward_concat = pd.concat([(~df['predicted_answer'] if 'neg' in k else df['predicted_answer']) for k, df in dfs.items() if 'forward' in k], axis = 1)
    backward_concat = pd.concat([(~df['predicted_answer'] if 'neg' in k else df['predicted_answer']) for k, df in dfs.items() if 'backward' in k], axis = 1)

    forward_maj = answer_majority(forward_concat)
    backward_maj = answer_majority(backward_concat)

    dfs['forward_maj'] = dfs['forward_pos'].copy()
    dfs['backward_maj'] = dfs['backward_pos'].copy()

    dfs['forward_maj']['predicted_answer'] = forward_maj
    dfs['backward_maj']['predicted_answer'] = backward_maj
    
    for k, df in dfs.items():
        df['key'] = k
        df['correct'] = (df['predicted_answer'] == df['gold_answer'])

    df_concat = pd.concat([df for df in dfs.values()], axis = 0)
    return pd.pivot_table(df_concat, index = 'key', columns = 'num_hops', values = 'correct', margins = True) * 100, df_concat

def evaluate_baseline_answer():
    """
    Returns a tuple with:
    (accuracy of baseline, concatenated DataFrame with CoT etc.)
    """
    dfs = {} # TODO: why is performance identical across seeds?
    for k, v in BASELINE.items():
        dfs[k] = pickle.load(open(ROOT + v, 'rb'))
        dfs[k]['predicted_answer'] = dfs[k]['predicted_answer'].replace({'True': True, 'False': False})
        dfs[k]['gold_answer'] = dfs[k]['gold_answer'].replace({'True': True, 'False': False})

    for k, df in dfs.items():
        df['key'] = k
        df['correct'] = (df['predicted_answer'] == df['gold_answer'])

    df_concat = pd.concat([df for df in dfs.values()], axis = 0)
    return pd.pivot_table(df_concat, index = 'key', columns = 'num_hops', values = 'correct', margins = True) * 100, df_concat
