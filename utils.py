import random

import dsp


def negate_query(sentence):
    words = sentence.split()
    if "not" in words:
        words = [word for word in words if word != "not"]
    elif "is" in words:
        ind = words.index("is")
        words.insert(ind + 1, "not")
    elif "are" in sentence:
        ind = words.index("are")
        words.insert(ind + 1, "not")
    else:
        raise ValueError(sentence)
    return " ".join(words)


def reverse_sentences(text):
    # Reverse the order of sentences in this string
    if text[-1] == '.':
        text = text[:-1]
    text = text.split('. ')[::-1]
    text = '. '.join(text).strip()
    text += "."
    return text


def flip_conclusion_in_cot(proof, negated_gold_cot_conclusion):
    if proof[-1] == ".":
        proof = proof[:-1]

    # Flip the final conclusion in chain-of-thought
    proof = proof.split('. ')
    negated_gold_cot_conclusion = negated_gold_cot_conclusion[:-1] # remove period

    idx = [i for i, p in enumerate(proof) if negated_gold_cot_conclusion in p][0]

    proof[idx] = negate_query(proof[idx])
        
    proof = '. '.join(proof).strip()
    proof += "."
    return proof


def remove_answer_from_proof(proof):
    if proof[-1] == '.':
        proof = proof[:-1]
    proof = proof.split('. ')

    while "True" in proof:
        proof.remove("True")
    
    while "False" in proof:
        proof.remove("False")

    proof = '. '.join(proof).strip()
    proof += "."

    return proof


def get_all_cot_steps(cots, list=False):
    # Remove the last period of the cot if exists
    cots_ = []
    for cot in cots:
        if cot == "": # ignore empty strings
            continue
        if cot[-1] == '.':
            cots_.append(cot[:-1])
        else:
            cots_.append(cot)
    
    # Separate into steps
    if not list:
        cot_steps = [set(cot.split('. ')) for cot in cots_]
    else:
        cot_steps = [cot.split('. ') for cot in cots_]

    # Remove empty strings
    while '' in cot_steps:
        cot_steps.remove('')

    return cot_steps


def get_demos_backward_cot(df, id):
    examples = []
    for ex_num in range(8):
        demo_dict = df.iloc[id][f"in_context_example{ex_num}"]
        facts_and_rules = demo_dict["question"]
        query = demo_dict["query"]
        proof = " ".join(demo_dict["chain_of_thought"][::-1])
        answer = demo_dict["answer"]

        ex = dsp.Example(
            facts_and_rules=facts_and_rules,
            query=query,
            proof=proof,
            answer=answer,
        )
        examples.append(ex)
    return examples


def get_demos_forward_cot(df, id):
    examples = []
    for ex_num in range(8):
        demo_dict = df.iloc[id][f"in_context_example{ex_num}"]
        facts_and_rules = demo_dict["question"]
        query = demo_dict["query"]
        proof = " ".join(demo_dict["chain_of_thought"])
        answer = demo_dict["answer"]

        ex = dsp.Example(
            facts_and_rules=facts_and_rules,
            query=query,
            proof=proof,
            answer=answer,
        )
        examples.append(ex)
    return examples


def get_test_example_cot(df, id, negate, random_order, get_demos, k):
    demo_dict = df.iloc[id][f"test_example"]
    facts_and_rules = demo_dict["question"]

    if random_order:
        facts_and_rules = randomize_order(facts_and_rules)

    query = demo_dict["query"]

    if negate:
        query = negate_query(query)

    demos = []
    if k > 0:
        demos = get_demos(df, id)
        demos = dsp.sample(demos, k=k)

    return dsp.Example(
        facts_and_rules=facts_and_rules, query=query, demos=demos
    )


def get_test_answer_forward_cot(df, id, negate=False):
    ex_dict = df.iloc[id][f"test_example"]
    answer = ex_dict["answer"]
    cot = ex_dict["chain_of_thought"]

    if negate:
        if answer == "True":
            answer = "False"
        else:
            answer = "True"

    return " ".join(cot), answer


def get_test_answer_backward_cot(df, id, negate=False):
    ex_dict = df.iloc[id][f"test_example"]
    answer = ex_dict["answer"]
    cot = ex_dict["chain_of_thought"][::-1]

    if negate:
        if answer == "True":
            answer = "False"
        else:
            answer = "True"

    return " ".join(cot), answer


def randomize_order(question):
    parts = question.split('. ')
    random.shuffle(parts)
    new_question = ". ".join(parts).replace('..', '.')

    return new_question


def print_template_example(
    df, id, template, get_demos, get_test_example, negate, random_order, seed, k
):
    ex = get_test_example(
        df, id, negate, random_order, get_demos=get_demos, k=k
    )

    print(template(ex))