import argparse
import pandas as pd
import pickle
import os
import dsp
from tqdm import tqdm

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
        template = prompts.forward_template
        get_demos = utils.get_demos_forward_cot
        get_test_answer = utils.get_test_answer_forward_cot
    elif args.prompt_type == "backward":
        template = prompts.backward_template
        get_demos = utils.get_demos_backward_cot
        get_test_answer = utils.get_test_answer_backward_cot
    elif args.prompt_type == "baseline":
        template = prompts.baseline_template
        get_demos = utils.get_demos_forward_cot
        get_test_answer = utils.get_test_answer_forward_cot
    else:
        raise NotImplementedError(f"prompt type {args.prompt_type} not implemented!")

    get_test_example = utils.get_test_example_cot

    return template, get_demos, get_test_answer, get_test_example


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_type", type=str, default="forward", choices=PROMPT_TYPES
    )
    parser.add_argument(
        "--negate",
        action="store_true",
        help="whether to negate query",
        default=False,
    )
    parser.add_argument(
        "--randomized_order",
        action="store_true",
        help="whether to randomize order of facts and rules",
    )
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--dataset", type=str, default="prontoqa_fictional")
    parser.add_argument(
        "--data_file",
        type=str,
        default="prontoqa_data/fictional/sampled_data.pkl",
    )
    parser.add_argument(
        "--output_dir", type=str, default="prontoqa_output/fictional"
    )

    parser.add_argument("--openai_model", type=str, default="text-davinci-003")

    parser.add_argument("--test_mini_batch", action="store_true", help="whether to test forward pass on mini batch first")

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
    seed,
    verbose=False,
):
    # return df of [id, num_hops, test_example, predicted_cot, predicted_answer, gold_cot, gold_answer]
    completions = []
    for id in tqdm(range(start, start + num_total)):
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
            seed=seed
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
            "test_example": df.iloc[id]["test_example"],
            "predicted_cot": predicted_cot,
            "predicted_answer": predicted_answer,
            "gold_cot": gold_cot,
            "gold_answer": gold_answer,
        }

        completions.append(data)

    out_df = pd.DataFrame(completions)

    with open(out_file, "wb") as f:
        pickle.dump(out_df, f)

    print(f"Dumped completions to {out_file}")
    return out_df


def main(args):
    with open(args.data_file, "rb") as f:
        data_df = pickle.load(f)

    filename = args.prompt_type
    if args.negated:
        filename += "_negated"
    if args.randomized_order:
        filename += "_randomized_order"

    filename += f"_seed_{args.seed}"
    out_file = os.path.join(args.output_dir, filename)

    template, get_demos, get_test_example, get_test_answer = get_functions(args)

    if args.test_mini_batch:
        num_total = 3
    else:
        num_total = len(data_df)

    out_df = sample_completion(
        start=0,
        num_total=num_total,
        k=0,
        df=data_df,
        template=template,
        get_demos=None,  # should be able to be None since k = 0, but if not, put get_demos
        get_test_example=get_test_example,
        get_test_answer=get_test_answer,
        out_file=out_file,
        negate=args.negate,
        random_order=args.randomized_order,
        seed=args.seed,
        verbose=False,
    )


if __name__ == "__main__":
    lm = dsp.GPT3(model="text-davinci-003", api_key=OPENAI_API_KEY)
    dsp.settings.configure(lm=lm)

    args = parse_args()
    main(args)
