import dsp

def baseline_template():
    Facts_And_Rules = dsp.Type(
    prefix="Facts and rules:", 
    desc="${the facts and rules}")

    Query = dsp.Type(
        prefix="Query:", 
        desc="${the query}")

    Proof = dsp.Type(
        prefix="Proof:", 
        desc="${a step-by-step proof that the query is true or false based only on the facts and rules}",
        format=dsp.format_answers
        )

    Answer = dsp.Type(
        prefix="Answer: Let's think step by step.", 
        desc="${the final answer based on the above proof}",
        format=dsp.format_answers
        )

    forward_cot_template = dsp.Template(
        instructions="Reason step-by-step over the facts and rules to determine whether the query is true or false",
        facts_and_rules=Facts_And_Rules(),
        query=Query(),
        proof=Proof(),
        answer=Answer()
        )
    
    return forward_cot_template