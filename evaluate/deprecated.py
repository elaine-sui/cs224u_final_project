
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