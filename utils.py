import random
import re
from copy import copy

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


def maybe_flip_conclusion_in_cot(proof, conclusion_should_have_not, query):
    # if the conclusion should have a not but the conclusion in the predicted cot doesn't have a not
    # if the conclusion should not have a not but the conclusion in the predicted cot has a not
    # then flip conclusion if in the predicted cot
    old_proof = copy(proof)
    if proof[-1] == ".":
        proof = proof[:-1]

    # Flip the final conclusion in chain-of-thought
    proof = proof.split('. ')

    negated_query = negate_query(query)

    if query in proof: # conclusion in pred cot is the same as the query
        if ("not" in query and not conclusion_should_have_not) \
            or ("not" not in query and conclusion_should_have_not):
            # flip conclusion (remove the not)
                idx = [i for i, p in enumerate(proof) if query in p][0]
        else:
            return old_proof
    elif negated_query in proof: # conclusion in pred cot is the same as the negated query
        if ("not" in negated_query and not conclusion_should_have_not) \
            or ("not" not in negated_query and conclusion_should_have_not):
            # flip conclusion (remove the not)
                idx = [i for i, p in enumerate(proof) if negated_query in p][0]
        else:
            return old_proof
    else:
        return old_proof

    proof[idx] = negate_query(proof[idx])   
    proof = '. '.join(proof).strip()
    proof += "."
    return proof


def get_all_cot_steps(cots, save_duplicates=False):
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
    if not save_duplicates:
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


def clean_cot_reasoning(cot):
    # Remove all occurrences of "Therefore, ", double negatives from predicted_cots, etc.
    clean_cot = re.sub("Therefore, ", "", cot)
    clean_cot = re.sub("Therefore, ", "", clean_cot)
    clean_cot = re.sub("However, ", "", clean_cot)
    clean_cot = re.sub("However ", "", clean_cot)
    clean_cot = re.sub("not not ", "", clean_cot)
    clean_cot = re.sub("Since ", "", clean_cot)
    clean_cot = re.sub("But ", "", clean_cot)
    clean_cot = re.sub("but ", "", clean_cot)
    clean_cot = re.sub(" and ", ". ", clean_cot)
    clean_cot = re.sub(",", ". ", clean_cot)
    clean_cot = re.sub("False.", "", clean_cot)
    clean_cot = re.sub("True.", "", clean_cot)
    clean_cot = re.sub("  ", " ", clean_cot)

    # Dealing with "or" ("Alex is not a numpus or a jompus" --> Alex is not a numpus; Alex is not a jompus
    sentences_with_or = re.findall(
        r"([A-Z][a-z\s]*) or ([A-Za-z\s]*).", clean_cot
    )

    if len(sentences_with_or) > 0:
        for sentence in sentences_with_or:
            intro_idx_end = sentence[0].index(" a ")
            intro = sentence[0][:intro_idx_end]
            parts = [sentence[0], intro + " " + sentence[1]]
            combined = ". ".join(parts) + "."

            # add combined sentence at the end
            clean_cot += combined

        # remove all "or" sentences
        if clean_cot[-1] == ".":
            clean_cot = clean_cot[:-1]

        cot_steps = clean_cot.split(". ")
        for s in cot_steps:
            if " or " in s:
                cot_steps.remove(s)

        # recombine
        clean_cot = ". ".join(cot_steps) + "."
    
    return clean_cot.strip()

def clean_answer(answer):
    # Remove periods from answers
    if answer[-1] == ".":
        answer = answer[:-1]
    return answer