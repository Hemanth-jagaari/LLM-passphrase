import os

# os.environ["HF_HOME"] = "../hf_home"

import pickle

import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

import pw_utils
from args_parser import parse_arguments
from model_manager import load_model_and_tokenizer
from prompt_manager import create_convs


def process_out(input_toks, tokenizer, generated_toks, out_ents):

    EOS = tokenizer.eos_token_id

    bsz, in_len = input_toks.shape

    out = generated_toks[:, in_len:]

    eos_n_1 = ((out == EOS).cumsum(1).cumsum(1) == 1).to(torch.uint8).argsort(1)[
        :, -1
    ] + 1

    return [
        (
            row[:ind].tolist(),
            ents_row[:ind].tolist(),
            tokenizer.decode(row[:ind]),
        )
        for row, ents_row, ind in zip(out, out_ents, eos_n_1)
    ]


@torch.inference_mode()
def generate_toks(
    input_toks,
    model,
    tokenizer,
    red_list=None,
    end_token_ids=None,
    max_num_toks=50,
    temperature=1.0,
    bot_q=0.0,
    top_p=1.0,
    seed=None,
):

    EOS = tokenizer.eos_token_id

    torch.manual_seed(seed)
    bsz, prompt_len = input_toks.shape
    eos_flag = torch.zeros(bsz, 1, dtype=torch.bool)
    out_ents = torch.empty(bsz, 0).to("cuda")

    for inx in range(max_num_toks):

        logits = model(input_toks).logits[:, -1, :]

        if red_list:
            logits[:, red_list] = -float("inf")

        probs = F.softmax(logits / temperature, dim=-1)

        nan_check = torch.isnan(probs)
        if (nan_check).any():
            rows_with_nan = nan_check.any(dim=1)
            rows_with_nan_indices = torch.nonzero(rows_with_nan).squeeze()

            if rows_with_nan_indices.dim() == 0:
                rows_with_nan_indices = rows_with_nan_indices.unsqueeze(0)

            for idx in rows_with_nan_indices:
                probs[idx] = 0
                probs[idx, EOS] = 1.0

        next_tok, next_tok_ent = pw_utils.pw_sample_top_p(
            probs, bot_q, top_p, EOS, end_token_ids
        )

        input_toks = torch.cat((input_toks, next_tok), dim=1)
        out_ents = torch.cat((out_ents, next_tok_ent), dim=1)

        if (next_tok == EOS).any():
            eos_flag[next_tok == EOS] = True

        if eos_flag.all():
            break

    return input_toks, out_ents


def main():

    args = parse_arguments()

    model_name, model, tokenizer = load_model_and_tokenizer(args.model)
    print("---- using model", model_name)

    red_list = None
    if args.red_list:
        with open(
            "./red_lists/" + model_name + "_red_list.pkl",
            "rb",
        ) as f:
            red_list = pickle.load(f)

    end_token_ids = None
    if args.end_token_ids:
        with open("./red_lists/" + model_name + "_end_token_list.pkl", "rb") as f:
            end_token_ids = pickle.load(f)

    convs, quote_label = create_convs(args.in_prompt)

    sd = args.sd

    bsz = args.bsz

    if args.out_subdir is None:
        out_dir = f"./output_dir/{model_name}_{os.path.splitext(args.in_prompt)[0]}"
    else:
        out_dir = f"./output_dir/{args.out_subdir}"

    prt_flag = False

    temp_list = [1.0, 1.2, 1.4]
    if args.temp_list:
        temp_list = args.temp_list

    top_p_list = [0.95, 0.99, 1.0]
    if args.top_p_list:
        top_p_list = args.top_p_list

    bot_q = args.bot_q
    t_p_grid = [(t, p) for t in temp_list for p in top_p_list]
    N = args.N

    for inx, conv in enumerate(convs):
        if not args.in_prompt.isdigit():
            quote_label = conv[-1]["content"]

        input_toks = tokenizer.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        input_toks = input_toks.repeat(bsz, 1)

        for t_p in t_p_grid:
            temperature, top_p = t_p

            for inx in range(N):
                generated_toks, out_ents = generate_toks(
                    input_toks,
                    model,
                    tokenizer,
                    red_list,
                    end_token_ids,
                    max_num_toks=args.max_len,
                    temperature=temperature,
                    bot_q=bot_q,
                    top_p=top_p,
                    seed=sd + inx,
                )
                out = process_out(input_toks, tokenizer, generated_toks, out_ents)

                pw_utils.logger(
                    tokenizer,
                    out_dir,
                    model_name,
                    in_str=quote_label,
                    toks_ents_str=out,
                    temperature=temperature,
                    top_p=top_p,
                    bot_q=bot_q,
                )

        pw_utils.log_args(
            out_dir,
            model_name,
            quote_label,
            args,
            temp_list,
            top_p_list,
        )


if __name__ == "__main__":
    main()
