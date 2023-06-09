import dsp


@dsp.transformation
def generic_dsp(
    df,
    id,
    template,
    get_demos,
    get_test_example,
    k=2,
    temperature=0.0,
    negate=False,
    random_order=False,
    seed=1234,
):
    example = get_test_example(df, id, negate, random_order, get_demos, k)

    if k > 0:
        demos = get_demos(df, id)
        example.demos = dsp.sample(demos, k=k)
    else:
        example.demos = []

    # Run your program using `template`:
    example, example_compl = dsp.generate(template, temperature=temperature)(
        example, stage="cot"
    )

    # Return the `dsp.Completions`:
    return example_compl
