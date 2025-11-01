from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import os
import torch


def _get_hf_token():
    """Return a Hugging Face auth token if set via env vars.

    Checks common variable names: HUGGINGFACE_HUB_TOKEN, HF_TOKEN, HF_AUTH_TOKEN.
    Returns None if not present (so public models still work).
    """
    for k in ["HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HF_AUTH_TOKEN"]:
        val = os.environ.get(k)
        if val:
            return val.strip()
    return None


def load_model_and_tokenizer(args_model):
    hf_token = _get_hf_token()

    force_cpu = os.environ.get("LLM_FORCE_CPU") == "1"
    # Allow user to turn on 4-bit quant via env var (when bitsandbytes installed)
    use_4bit = os.environ.get("LLM_USE_4BIT") == "1"
    offload_folder = os.environ.get("LLM_OFFLOAD_FOLDER")  # optional disk offload folder

    # device_map strategy
    if force_cpu:
        device_map = None
    else:
        device_map = "auto"

    # dtype preference: prefer bfloat16 when model paths specify it; else float16 unless on pure CPU
    def pick_dtype(pref):
        if force_cpu:
            return torch.float32  # safest for CPU
        return pref

    quantization_config = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig as _BnB
            quantization_config = _BnB(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        except Exception:
            quantization_config = None

    if args_model == "Llama-2-7b":
        model_path = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                dtype=pick_dtype(torch.float16),
                quantization_config=quantization_config,
                token=hf_token,
            )
        except Exception as e:
            print(f"[WARN] Primary load failed for {model_path}: {e}\nFalling back to CPU fp32 load.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=None,
                dtype=torch.float32,
                token=hf_token,
            )
    elif args_model == "Llama-2-13b":
        model_path = "meta-llama/Llama-2-13b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                dtype=pick_dtype(torch.float16),
                quantization_config=quantization_config,
                token=hf_token,
            )
        except Exception as e:
            print(f"[WARN] Primary load failed for {model_path}: {e}\nFalling back to CPU fp32 load.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=None,
                dtype=torch.float32,
                token=hf_token,
            )
    elif args_model == "Llama-2-70b":
        model_path = "meta-llama/Llama-2-70b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        # Force quantization for very large model unless force_cpu is set
        big_quant_cfg = BitsAndBytesConfig(
            llm_int4_threshold=200.0,
            bnb_4bit_compute_dtype=torch.float16,
        ) if not force_cpu else None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                dtype=pick_dtype(torch.float16),
                quantization_config=big_quant_cfg,
                token=hf_token,
            ).eval()
        except Exception as e:
            print(f"[WARN] Primary load failed for {model_path}: {e}\nAttempting CPU fp32 fallback (slow).")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=None,
                dtype=torch.float32,
                token=hf_token,
            ).eval()
    elif args_model == "Llama-3-8B":
        model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                dtype=pick_dtype(torch.bfloat16),
                quantization_config=quantization_config,
                token=hf_token,
            )
        except Exception as e:
            print(f"[WARN] Primary load failed for {model_path}: {e}\nFalling back to CPU fp32 load.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=None,
                dtype=torch.float32,
                token=hf_token,
            )
    elif args_model == "Llama-3-70B":
        model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                dtype=pick_dtype(torch.bfloat16),
                quantization_config=quantization_config,
                token=hf_token,
            )
        except Exception as e:
            print(f"[WARN] Primary load failed for {model_path}: {e}\nFalling back to CPU fp32 load.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=None,
                dtype=torch.float32,
                token=hf_token,
            )
    elif args_model == "gemma-7b":
        model_path = "google/gemma-7b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                dtype=pick_dtype(torch.bfloat16),
                quantization_config=quantization_config,
                token=hf_token,
            )
        except Exception as e:
            print(f"[WARN] Primary load failed for {model_path}: {e}\nFalling back to CPU fp32 load.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=None,
                dtype=torch.float32,
                token=hf_token,
            )
    elif args_model == "Mistral-7B":
        model_path = "mistralai/Mistral-7B-Instruct-v0.2"  
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                dtype=pick_dtype(torch.float16),
                quantization_config=quantization_config,
                token=hf_token,
            )
        except Exception as e:
            print(f"[WARN] Primary load failed for {model_path}: {e}\nFalling back to CPU fp32 load.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=None,
                dtype=torch.float32,
                token=hf_token,
            )
    elif args_model == "Mixtral-8x7B":
        model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1" 
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        big_quant_cfg = BitsAndBytesConfig(
            llm_int4_threshold=200.0,
            bnb_4bit_compute_dtype=torch.float16,
        ) if not force_cpu else None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                dtype=pick_dtype(torch.float16),
                quantization_config=big_quant_cfg,
                token=hf_token,
            ).eval()
        except Exception as e:
            print(f"[WARN] Primary load failed for {model_path}: {e}\nAttempting CPU fp32 fallback (slow).")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=None,
                dtype=torch.float32,
                token=hf_token,
            ).eval()
    else:
        raise ValueError(f"Unknown model: {args_model}")

    model_name = os.path.basename(model_path.rstrip("/"))

    # Optional disk offload hint
    if offload_folder and not force_cpu:
        try:
            from accelerate import disk_offload
            disk_offload(model, offload_folder=offload_folder)
        except Exception:
            pass

    # Debug summary
    try:
        print("[LOAD SUMMARY] model=", model_name,
              " force_cpu=", force_cpu,
              " use_4bit=", use_4bit,
              " quant_cfg=", bool(quantization_config) if 'quantization_config' in locals() else 'n/a',
              " device_map=", device_map)
    except Exception:
        pass

    return model_name, model, tokenizer
