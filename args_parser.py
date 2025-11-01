import argparse


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Generate text from selected language model."
    )
    parser.add_argument(
        "--in_prompt",
        default=6,
        help="enter either 1,2,3,... as number of examples in conv or enter input file as string",
    )
    parser.add_argument(
        "--out_subdir",
        type=str,
        default="test",
        help="out_subdir name (not whole path)",
    )
    parser.add_argument(
        "--bsz",
        type=int,
        default=16,
        help="batch size. On 2 A5000s, 16 works for all.",
    )
    parser.add_argument("--temp_list", type=float, nargs="+", help="list of temps")
    parser.add_argument("--top_p_list", type=float, nargs="+", help="list of top_ps")
    parser.add_argument(
        "--bot_q",
        type=float,
        default=0.0,
        help="this is top-p truncation value",
    )
    parser.add_argument("--N", type=int, default=1, help="number of iterations")
    parser.add_argument("--sd", type=int, default=100, help="random seed")
    parser.add_argument("--red_list", action="store_true", help="red_list")
    parser.add_argument("--end_token_ids", action="store_true", help="end_token_ids")
    parser.add_argument(
        "--max_len",
        type=int,
        default=50,
        help="maximum number of generation tokens (default 50)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "Llama-2-7b",
            "Llama-2-13b",
            "Llama-2-70b",
            "Llama-3-8B",
            "Llama-3-70B",
            "gemma-7b",
            "Mistral-7B",
            "Mixtral-8x7B",
        ],
        default="Llama-2-7b",
        help="Choose a language model among Llama-2-7b, Llama-2-13b, Llama-2-70b, Llama-3-8B, Llama-3-70B, Mistral-7B, Mixtral-8x7B, gemma-7b. Default is 'Llama-2-7b' when no arg is provided.",
    )
    return parser.parse_args()
