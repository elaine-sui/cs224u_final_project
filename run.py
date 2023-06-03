import argparse
import pandas as pd
import pickle
import os
import dsp
from tqdm import tqdm
import numpy as np
import random

from dsp_program import generic_dsp
import utils
import prompts


root_path = "."
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(root_path, "cache")
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)  # or replace with your API key (optional)

PROMPT_TYPES = ["forward", "backward", "baseline"]


def get_functions(args):
    if args.prompt_type == "forward":
        template = prompts.forward_template()
        get_demos = utils.get_demos_forward_cot
        get_test_answer = utils.get_test_answer_forward_cot
    elif args.prompt_type == "backward":
        template = prompts.backward_template()
        get_demos = utils.get_demos_backward_cot
        get_test_answer = utils.get_test_answer_backward_cot
    elif args.prompt_type == "baseline":
        template = prompts.baseline_template()
        get_demos = utils.get_demos_forward_cot
        get_test_answer = utils.get_test_answer_forward_cot
    else:
        raise NotImplementedError(
            f"prompt type {args.prompt_type} not implemented!"
        )

    get_test_example = utils.get_test_example_cot

    return template, get_demos, get_test_answer, get_test_example


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-type",
        "-pt",
        type=str,
        default="forward",
        choices=PROMPT_TYPES,
    )
    parser.add_argument(
        "--negate",
        "-n",
        action="store_true",
        help="whether to negate query",
        default=False,
    )
    parser.add_argument(
        "--randomized-order",
        action="store_true",
        help="whether to randomize order of facts and rules",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed. Pass 0 for no seed"
    )
    parser.add_argument("--dataset", type=str, default="prontoqa_fictional")
    parser.add_argument(
        "--data-file",
        type=str,
        default="prontoqa_data/fictional/sampled_data.pkl",
    )
    parser.add_argument(
        "--output-dir", type=str, default="prontoqa_output/fictional"
    )

    parser.add_argument("--openai_model", type=str, default="text-davinci-003")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument(
        "--test-mini-batch",
        "-tmb",
        action="store_true",
        help="whether to test forward pass on mini batch first",
    )

    parser.add_argument(
        "--k", type=int, default=0, help="number of few-shot examples"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    assert os.path.exists(args.data_file)

    return args


def sample_completion(
    start,
    num_total,
    k,
    df,
    template,
    get_demos,
    get_test_example,
    get_test_answer,
    out_file,
    negate,
    random_order,
    temperature,
    verbose=False,
):
    # return df of [id, num_hops, test_example, predicted_cot, predicted_answer, gold_cot, gold_answer]
    completions = []
    for i, id in enumerate(tqdm(range(start, start + num_total))):
        if verbose:
            print(id)

        completion = generic_dsp(
            df,
            id,
            template=template,
            k=k,
            get_demos=get_demos,
            get_test_example=get_test_example,
            negate=negate,
            random_order=random_order,
            temperature=temperature,
        )

        predicted_answer = completion.answer
        predicted_cot = completion.proof

        gold_cot, gold_answer = get_test_answer(df, id, negate)

        if verbose:
            print(f"Query: {completion.query}")
            print(f"Predicted COT: {predicted_cot}")
            print(f"Gold COT: {gold_cot}")

            print(f"Predicted answer: {predicted_answer}")
            print(f"Gold answer: {gold_answer}")
            print("=" * 80)

        data = {
            "id": df.iloc[id]["id"],
            "num_hops": df.iloc[id]["num_hops"],
            "test_example": df.iloc[id][
                "test_example"
            ],  # note that this is the original question and not negated or with randomized fact order.
            "predicted_cot": predicted_cot,
            "predicted_answer": predicted_answer,
            "gold_cot": gold_cot,
            "gold_answer": gold_answer,
        }

        completions.append(data)

        # Batch save
        if i > 0 and i % 50 == 0:
            out_file_temp = out_file[:-4] + f"_{i}.pkl"
            out_df = pd.DataFrame(completions)
            with open(out_file_temp, "wb") as f:
                pickle.dump(out_df, f)

            print(f"Dumped partial (i={i}) completions to {out_file_temp}")

    out_df = pd.DataFrame(completions)

    with open(out_file, "wb") as f:
        pickle.dump(out_df, f)

    print(f"Dumped completions to {out_file}")
    return out_df


def main(args):
    with open(args.data_file, "rb") as f:
        data_df = pickle.load(f)

    # Set seed
    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)

    data_df = data_df.reset_index(drop=True)

    filename = args.prompt_type
    if args.negate:
        filename += "_negated"
    if args.randomized_order:
        filename += "_randomized_order"

    filename += f"_{args.k}_shot"

    filename += f"_temp_{args.temperature}"

    # note: seed is really only used for randomizing order of facts/rules.
    # but it also serves to differentiate between different runs
    filename += f"_seed_{args.seed}"
    out_file = os.path.join(args.output_dir, filename + ".pkl")

    template, get_demos, get_test_answer, get_test_example = get_functions(args)

    utils.print_template_example(
        data_df,
        0,
        template,
        get_demos,
        get_test_example,
        args.negate,
        args.randomized_order,
        args.seed,
        k=args.k,
    )

    if args.test_mini_batch:
        num_total = 1
        data_df = (
            data_df[data_df["num_hops"] == 5]
            .sample(frac=1)
            .reset_index(drop=True)
        )
    else:
        num_total = len(data_df)

    out_df = sample_completion(
        start=0,
        num_total=num_total,
        k=args.k,
        df=data_df,
        template=template,
        get_demos=get_demos,
        get_test_example=get_test_example,
        get_test_answer=get_test_answer,
        out_file=out_file,
        negate=args.negate,
        random_order=args.randomized_order,
        temperature=args.temperature,
        verbose=False,
    )


if __name__ == "__main__":
    lm = dsp.GPT3(model="text-davinci-003", api_key=OPENAI_API_KEY)
    dsp.settings.configure(lm=lm)

    args = parse_args()
    main(args)
