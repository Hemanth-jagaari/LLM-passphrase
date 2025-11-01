# LLM-passphrase

Generate short, story-like passphrases using Large Language Models (LLMs). The code composes chat-style prompts, samples model outputs with configurable temperature / top-p / q-cutoff, and logs token-level entropies.

---
## 1. Quick Start

```bash
git clone https://github.com/JieSLi/LLM-passphrase.git
cd LLM-passphrase

# (Recommended) Create a virtual environment (macOS / Linux)
python -m venv .venv
source .venv/bin/activate

# (Windows PowerShell)
python -m venv .venv
./.venv/Scripts/Activate.ps1

# (Windows CMD)
python -m venv .venv
\.venv\Scripts\activate.bat

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# First tiny test run (downloads model if not cached)
python main_sample.py --model gemma-7b --in_prompt 6 --bsz 1 --N 1 --out_subdir quick_test
```

The first run will download the chosen model (could be multiple GB). Subsequent runs reuse the cache.

If you encounter SSL or permission issues on Windows, see the Windows section below.

---
## 2. Supported Models & Identifiers

| Arg value       | Hugging Face repo slug                          | Precision / Notes                  |
|-----------------|--------------------------------------------------|------------------------------------|
| Llama-2-7b      | meta-llama/Llama-2-7b-chat-hf                    | fp16                               |
| Llama-2-13b     | meta-llama/Llama-2-13b-chat-hf                   | fp16                               |
| `--max_len`     | Max generation tokens (default 50)               |
| Llama-2-70b     | meta-llama/Llama-2-70b-chat-hf                   | 4-bit (bitsandbytes) quant         |
| Llama-3-8B      | meta-llama/Meta-Llama-3-8B-Instruct              | bfloat16                           |
| Llama-3-70B     | meta-llama/Meta-Llama-3-70B-Instruct             | bfloat16                           |
| gemma-7b        | google/gemma-7b-it                              | bfloat16                           |
| Mistral-7B      | mistralai/Mistral-7B-Instruct-v0.2              | fp16                               |
| Mixtral-8x7B    | mistralai/Mixtral-8x7B-Instruct-v0.1            | 4-bit (bitsandbytes) quant         |

> Larger models (13B, 70B, Mixtral) require significant GPU VRAM. Start with `gemma-7b` or `Mistral-7B` if testing locally on limited hardware.

---
## 3. Model Download Methods

### A. Automatic (Recommended)
Simply run `main_sample.py`; `transformers` will fetch weights into the default cache (`~/.cache/huggingface/` or `${HF_HOME}` if set).

### B. Prefetch via `huggingface-cli`
```bash
pip install huggingface_hub
huggingface-cli login        # if the model requires gated access
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./models/Llama-2-7b
```
Then run with environment variable pointing to cache (optional):
```bash
export HF_HOME=$(pwd)/models
python main_sample.py --model Llama-2-7b --in_prompt 6 --bsz 1 --N 1
```

### C. Programmatic Prefetch
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/gemma-7b-it", local_dir="./models/gemma-7b-it")
```

### D. Using Git LFS (when necessary)
Some repos (e.g., older ones) can be cloned directly:
```bash
git lfs install
git clone https://huggingface.co/google/gemma-7b-it
```

### Disk Space & Cache
Set a custom cache dir if you need space management:
```bash
export HF_HOME=/Volumes/External/hf_cache
```
Clean unused refs:
```bash
rm -rf ~/.cache/huggingface/*/*/snapshots/<old-sha>
```

### Gated / Restricted Models (Gemma, Llama)
If you see an error like:
```
Cannot access gated repo ... Access to model ... is restricted.
```
Do this:
1. Visit the model page (e.g. https://huggingface.co/google/gemma-7b-it) and click "Access" / accept license terms.
2. Log in locally: `huggingface-cli login` (opens browser or asks for token).
3. Export your token (optional if already logged in, required for non-interactive scripts):
  * macOS/Linux: `export HUGGINGFACE_HUB_TOKEN=hf_xxxxx`
  * PowerShell: `$Env:HUGGINGFACE_HUB_TOKEN = "hf_xxxxx"`
4. Re-run the script. The code now passes `token=` automatically when set.
5. If still failing, verify you used the correct repo slug (see table) and that access was granted (check your HF profile → settings → tokens).

Temporary workaround (while waiting for approval): use an open model:
```bash
python main_sample.py --model Mistral-7B --in_prompt 6 --bsz 1 --N 1 --out_subdir alt_test
```

---
## 4. Running Experiments

### Basic Example (Economics prompt set 806, Gemma 7B)
```bash
python main_sample.py --model gemma-7b --out_subdir econ_exs --bsz 2 --N 4 --in_prompt 806
```

### Grid Over Temperature / Top-p
```bash
python main_sample.py --model Mistral-7B \
  --in_prompt 6 \
  --bsz 2 \
  --N 3 \
  --temp_list 0.8 1.0 1.2 \
  --top_p_list 0.9 0.95 1.0 \
  --out_subdir mistral_grid
```

### Red List / End Token Lists (if files present)
```bash
python main_sample.py --model gemma-7b --in_prompt 6 --red_list --end_token_ids
```
Expected to look for: `./red_lists/<model_name>_red_list.pkl` and `_end_token_list.pkl`.

### Very Small Smoke Test
```bash
python main_sample.py --model gemma-7b --in_prompt 6 --bsz 1 --N 1 --temp_list 1.0 --top_p_list 1.0 --out_subdir smoke
```

Output JSON logs are written under `./output_dir/<out_subdir>/` with timestamped file names.

---
## 5. Prompts (`--in_prompt`)
Numeric IDs map to curated example sets (see `prompt_manager.py`):

| Range  | Category                | Max examples (approx) | Label prefix in logs |
|--------|-------------------------|------------------------|----------------------|
| <100   | Base examples           | variable               | `<n>_exs`            |
| 100-199| Random words            |                        | `rand_<n>_exs`       |
| 200-299| Low entropy             |                        | `loent_<n>_exs`      |
| 300-399| High entropy            |                        | `hient_<n>_exs`      |
| 400-499| Nouns & verbs           |                        | `nounV_<n>_exs`      |
| 500-599| Adjectives & nouns      |                        | `adjN_<n>_exs`       |
| 600-699| Cats theme              |                        | `cats_<n>_exs`       |
| 700-799| Science                 |                        | `sci_<n>_exs`        |
| 800-899| Economics               |                        | `econ_<n>_exs`       |

If you pass a string that is not a digit or is ≥ 900, it treats it as a filename inside `./input_dir/` whose lines become prompts.

---
## 6. Arguments Summary
Run `python args_parser.py -h` or consult below:

| Flag | Purpose |
|------|---------|
| `--in_prompt` | Numeric prompt set or input filename |
| `--out_subdir` | Output folder name under `output_dir/` |
| `--bsz` | Batch size (number of parallel generations) |
| `--temp_list` | List of temperatures (grid with top-p) |
| `--top_p_list` | List of top-p values |
| `--bot_q` | Lower probability cutoff (q) for nucleus variant |
| `--N` | Iterations per (temp, top-p) combination |
| `--sd` | Base random seed |
| `--red_list` | Use per-model token exclusion list |
| `--end_token_ids` | Treat provided tokens as EOS |
| `--model` | Select model from supported set |

Token sampling logic is implemented in `pw_utils.pw_sample_top_p`.

---
## 7. Performance & Memory Tips
1. Prefer smaller models locally (7B) unless you have ≥24GB GPU memory.
2. Reduce `--bsz` if you hit CUDA OOM errors.
3. Limit generation length by editing `max_num_toks` in `generate_toks` (default 50).
4. Use 4-bit quantized options (Llama-2-70b, Mixtral) already configured in `model_manager.py`.
5. For pure CPU fallback add: `CUDA_VISIBLE_DEVICES=""` before the command (will be slow).
6. New env fallbacks:
  * `LLM_FORCE_CPU=1` force full CPU load (disables automatic layer device mapping).
  * `LLM_USE_4BIT=1` attempt 4-bit quantization for supported models (requires bitsandbytes; silently ignored if unavailable).
  * `LLM_OFFLOAD_FOLDER=offload_weights` apply disk offload after load (create folder first; slow, last resort).
### Example: Resource-Constrained Runs
Force CPU and shorten generation length:
```bash
LLM_FORCE_CPU=1 python main_sample.py --model gemma-7b --in_prompt 6 --bsz 1 --N 1 --max_len 30 --out_subdir cpu_short
```
Attempt 4-bit quantization (if bitsandbytes installed):
```bash
LLM_USE_4BIT=1 python main_sample.py --model Mistral-7B --in_prompt 6 --bsz 2 --N 2 --out_subdir q4_try
```
Disk offload (slow):
```bash
mkdir -p offload_weights
LLM_OFFLOAD_FOLDER=offload_weights python main_sample.py --model Llama-2-13b --in_prompt 6 --bsz 1 --N 1 --out_subdir offload_test
```

---
## 8. Custom Cache & Auth
If a model is gated (e.g. Llama 2 / 3), request access on Hugging Face, then:
```bash
huggingface-cli login
```
Set token for scripts:
```bash
export HUGGINGFACE_HUB_TOKEN=hf_xxx
```
On Windows PowerShell:
```powershell
$Env:HUGGINGFACE_HUB_TOKEN = "hf_xxx"
```
On Windows CMD:
```cmd
set HUGGINGFACE_HUB_TOKEN=hf_xxx
```

---
## 9. Logging & Outputs
Each batch produces a JSON containing:
* Generated sentence
* Token IDs & per-token entropy
* Sum of entropies (`sent_ent`)
* Sampling params (temperature, top_p, bot_q)

An args summary file named: `<model>_<label>__args_log_<date>.txt` is also written.

---
## 10. Suggested Improvements
Consider adding:
* Smaller dummy model option (e.g. `distilgpt2`) for ultra-fast tests.
* Command-line flag for `max_num_toks`.
* `requirements.txt` (included) and `pyproject.toml` for reproducibility.
* Simple evaluation script to aggregate entropy stats.

PRs welcome!

---
### Offload Error Troubleshooting ("You are trying to offload the whole model to the disk")
This occurs when `device_map="auto"` cannot place any layers on GPU/RAM and would need full disk offload. Fix paths:
1. Use smaller model: `--model gemma-7b` or `Mistral-7B`.
2. Force CPU load: `LLM_FORCE_CPU=1 python main_sample.py ...`.
3. Try 4-bit quantization: `LLM_USE_4BIT=1`.
4. Provide explicit offload folder: `LLM_OFFLOAD_FOLDER=offload_weights` (slow).
5. Reduce batch/length: `--bsz 1 --max_len 25`.

Diagnostics:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import transformers, bitsandbytes; print('Transformers version', transformers.__version__)"
```
If fallback triggers, loader prints a `[WARN]` and then performs CPU fp32 load (slow but consistent).

---
## 11. License

MIT License. See `LICENSE` (add one if missing).

---
## 12. Citation
If you use this in research, please cite the repository URL.

---
## 13. Windows Setup & Notes

### A. Python & Environment
1. Install Python 3.10+ from https://www.python.org/downloads/ (check "Add Python to PATH").
2. Create & activate venv:
  * PowerShell: `python -m venv .venv; ./.venv/Scripts/Activate.ps1`
  * CMD: `python -m venv .venv && .\.venv\Scripts\activate.bat`
  * Deactivate with `deactivate`.

If activation is blocked by execution policy in PowerShell, run (once):
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### B. Environment Variables
Set `HF_HOME` (optional custom cache):
```powershell
$Env:HF_HOME = "D:\hf_cache"
```
```cmd
set HF_HOME=D:\hf_cache
```
Persist via System Properties → Environment Variables.

### C. bitsandbytes Considerations
`bitsandbytes` GPU quantization may fail on Windows native due to CUDA build differences. Options:
1. Use WSL2 (recommended): Install Ubuntu, setup CUDA per NVIDIA docs, then install requirements.
2. Skip models using quantization (avoid 70B / Mixtral) and test with `gemma-7b` or `Mistral-7B`.
3. Try CPU-only (slow) or use an NVIDIA GPU with a recent driver + CUDA toolkit matching PyTorch build.

If `bitsandbytes` import errors appear, you can still run non-quantized models; quantized paths in `model_manager.py` will fallback only if module loads. To disable quantization forcibly, edit the relevant branches and remove `quantization_config`.

### D. Git LFS
Install Git LFS for large model repos:
```powershell
winget install Git.Git
git lfs install
```

### E. Long Paths
If you see path length errors, enable long paths:
1. Run `regedit`, navigate to `Computer\\HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\FileSystem`.
2. Set `LongPathsEnabled` to `1`.
3. Or via PowerShell (admin): `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`.

### F. Hugging Face Auth
PowerShell token set:
```powershell
huggingface-cli login
```
Stores credential in user cache; no need to export token for interactive CLI usage.

### G. WSL2 Recommendation
For best compatibility (CUDA, bitsandbytes, tooling) use WSL2 Ubuntu and follow the Linux instructions in this README. Access repo from `/mnt/c/...` or clone directly inside the Linux filesystem.

### H. Sample Windows Command
PowerShell example minimal run:
```powershell
python main_sample.py --model gemma-7b --in_prompt 6 --bsz 1 --N 1 --out_subdir win_smoke
```

### I. Troubleshooting Quick Reference
| Issue | Symptom | Fix |
|-------|---------|-----|
| ExecutionPolicy | Cannot run Activate.ps1 | `Set-ExecutionPolicy RemoteSigned` |
| bitsandbytes error | ImportError / CUDA not found | Use WSL2 or smaller non-quant models |
| OOM | CUDA out of memory | Lower `--bsz`; use smaller model |
| SSL error | Download fails | `pip install certifi; python -m certifi` ensure trust store, or run under WSL2 |
| Slow download | Stalled at large shards | Check network; prefetch via `huggingface-cli download` |

---

