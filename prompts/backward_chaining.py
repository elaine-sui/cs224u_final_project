import dsp


def backward_template():
    Facts_And_Rules = dsp.Type(
        prefix="Facts and rules:", desc="${the facts and rules}"
    )

    Query = dsp.Type(prefix="Query:", desc="${the query}")

    Proof = dsp.Type(
        prefix="Proof:",
        desc="${a step-by-step proof that the query is true or false based only on the facts and rules}",
        format=dsp.format_answers,
    )

    Answer = dsp.Type(
        prefix="Answer:",
        desc="${the final answer based on the above proof}",
        format=dsp.format_answers,
    )

    backward_cot_template = dsp.Template(
        instructions="Use backward chaining to reason over the facts and rules to determine whether the query is true or false",
        facts_and_rules=Facts_And_Rules(),
        query=Query(),
        proof=Proof(),
        answer=Answer(),
    )

    return backward_cot_template


def backward_ltsbs_template():
    Facts_And_Rules = dsp.Type(
        prefix="Facts and rules:", desc="${the facts and rules}"
    )

    Query = dsp.Type(prefix="Query:", desc="${the query}")

    Proof = dsp.Type(
        prefix="Proof: Let's think step by step.",
        desc="${a step-by-step proof that the query is true or false based only on the facts and rules}",
        format=dsp.format_answers,
    )

    Answer = dsp.Type(
        prefix="Answer:",
        desc="${the final answer based on the above proof}",
        format=dsp.format_answers,
    )

    backward_cot_template = dsp.Template(
        instructions="Use backward chaining to reason step by step over the facts and rules to determine whether the query is true or false",
        facts_and_rules=Facts_And_Rules(),
        query=Query(),
        proof=Proof(),
        answer=Answer(),
    )

    return backward_cot_template
