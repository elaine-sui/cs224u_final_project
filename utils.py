import dsp

def get_demos_backward_cot(df, id):
    examples = []
    for ex_num in range(8):
        demo_dict = df.iloc[id][f'in_context_example{ex_num}']
        facts_and_rules = demo_dict['question']
        query = demo_dict['query']
        proof = ' '.join(demo_dict['chain_of_thought'][::-1])
        answer = demo_dict['answer']
        
        ex = dsp.Example(
            facts_and_rules=facts_and_rules,
            query=query,
            proof=proof,
            answer=answer
        )
        examples.append(ex)
    return examples

def get_demos_forward_cot(df, id):
    examples = []
    for ex_num in range(8):
        demo_dict = df.iloc[id][f'in_context_example{ex_num}']
        facts_and_rules = demo_dict['question']
        query = demo_dict['query']
        proof = ' '.join(demo_dict['chain_of_thought'])
        answer = demo_dict['answer']
        
        ex = dsp.Example(
            facts_and_rules=facts_and_rules,
            query=query,
            proof=proof,
            answer=answer
        )
        examples.append(ex)
    return examples

def get_test_example_cot(df, id):
    demo_dict = df.iloc[id][f'test_example']
    facts_and_rules = demo_dict['question'] 
    query = demo_dict['query']

    return dsp.Example(facts_and_rules=facts_and_rules, query=query)

def get_test_answer_forward_cot(df, id):
    ex_dict = df.iloc[id][f'test_example']
    answer = ex_dict['answer']
    cot = ex_dict['chain_of_thought']
    return ' '.join(cot), answer

def get_test_answer_backward_cot(df, id):
    ex_dict = df.iloc[id][f'test_example']
    answer = ex_dict['answer']
    cot = ex_dict['chain_of_thought'][::-1]
    return ' '.join(cot), answer
