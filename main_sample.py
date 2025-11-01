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
    """Generate tokens continuing from input_toks on the same device as model.

    This version no longer hard-codes CUDA; it infers the proper device from
    model/input_toks so CPU fallbacks work correctly.
    """
    EOS = tokenizer.eos_token_id

    torch.manual_seed(seed)
    device = input_toks.device
    bsz, _ = input_toks.shape
    eos_flag = torch.zeros(bsz, 1, dtype=torch.bool, device=device)
    out_ents = torch.empty(bsz, 0, device=device)

    for _ in range(max_num_toks):
        logits = model(input_toks).logits[:, -1, :]

        if red_list:
            logits[:, red_list] = -float("inf")

        probs = F.softmax(logits / temperature, dim=-1)

        nan_check = torch.isnan(probs)
        if nan_check.any():
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

    # Determine the device of the model's first parameter for token placement.
    try:
        model_param_device = next(model.parameters()).device
    except StopIteration:
        model_param_device = torch.device("cpu")

    for inx, conv in enumerate(convs):
        if not args.in_prompt.isdigit():
            quote_label = conv[-1]["content"]
        # Always initialize; we'll assign below.
        input_toks = None
        if getattr(tokenizer, "chat_template", None):
            try:
                input_toks = tokenizer.apply_chat_template(
                    conv, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(model_param_device)
            except Exception as e:
                # Fallback to plain prompt if template application fails (e.g., malformed template)
                print(f"[WARN] chat template failed: {e}. Falling back to plain prompt construction.")
        if input_toks is None:
            # Plain prompt fallback (non-chat models or failed template)
            plain_parts = []
            for msg in conv:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    plain_parts.append(f"User: {content}\n")
                else:
                    plain_parts.append(f"Assistant: {content}\n")
            plain_parts.append("Assistant:")  # cue for generation
            plain_prompt = "".join(plain_parts)
            input_toks = tokenizer.encode(plain_prompt, return_tensors="pt").to(model_param_device)

        if inx == 0:
            print(f"[INFO] Starting generation loop: bsz={bsz}, temps={temp_list}, top_ps={top_p_list}, N={N}, max_len={args.max_len}, device={model_param_device}")

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
            print(f"[INFO] Completed temp={temperature} top_p={top_p}")

        pw_utils.log_args(
            out_dir,
            model_name,
            quote_label,
            args,
            temp_list,
            top_p_list,
        )
    print(f"[INFO] Generation complete. Output files written to: {out_dir}")


if __name__ == "__main__":
    main()
